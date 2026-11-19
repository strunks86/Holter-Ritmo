using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;

// Exporta un examen H-Scribe a WFDB usando HW32ExamAPI.dll (x86).
// uso: export180.exe <carpeta_examen> <salida_sin_extension> [segundos=0:todo] [carpeta_dll] [start_s=0]
//
// `segundos` limita la ventana; 0 = hasta el final real del registro.
// El final NO se toma de GetSummary: en varios exámenes (p. ej. Exam_1448)
// Summary informa una duración parcial (4,3 h de 24,3 h). Se avanza de a 10 s
// y se corta cuando aparecen 60 s seguidos de muestras en cero o la API falla;
// después se recortan las muestras finales en cero.
class Export
{
    [DllImport("kernel32", CharSet = CharSet.Unicode, SetLastError = true)]
    static extern IntPtr LoadLibraryW(string path);
    [DllImport("kernel32", CharSet = CharSet.Ansi, SetLastError = true)]
    static extern IntPtr GetProcAddress(IntPtr mod, string name);
    [DllImport("kernel32", CharSet = CharSet.Unicode)]
    static extern bool SetDllDirectoryW(string path);

    [UnmanagedFunctionPointer(CallingConvention.ThisCall)]
    delegate int OpenExamFn(IntPtr self, [MarshalAs(UnmanagedType.LPWStr)] string path, IntPtr info);
    [UnmanagedFunctionPointer(CallingConvention.ThisCall)]
    delegate int WaveFn(IntPtr self, int a, int b, int c, double d, IntPtr list);
    [UnmanagedFunctionPointer(CallingConvention.ThisCall)]
    delegate int SummaryFn(IntPtr self, IntPtr summary);

    const int SPS = 180;
    const int REC = 18;   // bytes por muestra
    const int STEP_MS = 10000;
    const int ZERO_BLOCKS = 6;      // 60 s en cero = fin del registro
    const long MAX_MS = 72L * 3600 * 1000;   // tope de seguridad

    static void Main(string[] args)
    {
        if (args.Length < 2)
        {
            Console.WriteLine(
                "uso: export180.exe <carpeta_examen> <salida> [segundos, 0=todo] [carpeta_dll] [start_s]");
            Environment.Exit(1);
        }
        string exam = args[0];
        string outBase = args[1];
        int maxSec = args.Length > 2 ? int.Parse(args[2]) : 0;
        string dllDir = args.Length > 3 ? args[3]
            : @"G:\Mi unidad\Estudio Examenes\HF\Mortara Instrument Inc\ModalityMgr\Modalities\HScribe";
        double startSec = args.Length > 4
            ? double.Parse(args[4], System.Globalization.CultureInfo.InvariantCulture)
            : 0.0;
        if (startSec < 0) startSec = 0;
        long startMs = (long)Math.Round(startSec * 1000.0);

        SetDllDirectoryW(dllDir);
        IntPtr mod = LoadLibraryW(Path.Combine(dllDir, "HW32ExamAPI.dll"));
        if (mod == IntPtr.Zero) { Console.WriteLine("LoadLibrary error " + Marshal.GetLastWin32Error()); Environment.Exit(2); }
        var open = (OpenExamFn)Marshal.GetDelegateForFunctionPointer(
            GetProcAddress(mod, "?OpenExam@HW32ExamAPI@@QAEHPA_WPAUexam_info@@@Z"), typeof(OpenExamFn));
        var wave = (WaveFn)Marshal.GetDelegateForFunctionPointer(
            GetProcAddress(mod, "?GetWaveform@HW32ExamAPI@@QAEHHHHNPAUsample_list@@@Z"), typeof(WaveFn));
        var summary = (SummaryFn)Marshal.GetDelegateForFunctionPointer(
            GetProcAddress(mod, "?GetSummary@HW32ExamAPI@@QAEHPAUsummary_info@@@Z"), typeof(SummaryFn));

        IntPtr self = Marshal.AllocHGlobal(65536);
        for (int i = 0; i < 65536; i += 8) Marshal.WriteInt64(self, i, 0);
        IntPtr info = Marshal.AllocHGlobal(4096);
        for (int i = 0; i < 4096; i += 8) Marshal.WriteInt64(info, i, 0);
        int rc = open(self, exam, info);
        if (rc != 0) { Console.WriteLine("OpenExam " + rc); Environment.Exit(4); }

        IntPtr sum = Marshal.AllocHGlobal(4096);
        for (int i = 0; i < 4096; i += 8) Marshal.WriteInt64(sum, i, 0);
        summary(self, sum);
        // Campo 12 = duración en ms según Summary. Solo se informa: no delimita
        // la exportación porque en algunos exámenes viene truncado.
        int summaryMs = Marshal.ReadInt32(sum, 12);
        Console.WriteLine("duracion segun Summary: " + summaryMs + " ms = "
                          + (summaryMs / 3600000.0).ToString("F2") + " h");
        int summaryBeats = Marshal.ReadInt32(sum, 4);
        Console.WriteLine("latidos segun Summary: " + summaryBeats);
        if (maxSec < 0) maxSec = 0;

        long endLimit = MAX_MS;
        if (maxSec > 0) endLimit = Math.Min(MAX_MS, startMs + (long)maxSec * 1000);

        string datPath = outBase + ".dat";
        string heaPath = outBase + ".hea";
        string name = Path.GetFileName(outBase);
        int totalSamples = 0;
        int zeroBlocks = 0;
        List<short> buffer = new List<short>(1 << 20);

        IntPtr list = Marshal.AllocHGlobal(512);
        for (long t = startMs; t < endLimit; t += STEP_MS)
        {
            long blockEnd = Math.Min(t + STEP_MS, endLimit);
            for (int i = 0; i < 512; i += 8) Marshal.WriteInt64(list, i, 0);
            int n = 0;
            int ok = wave(self, (int)t, (int)blockEnd, SPS, 1.0, list);
            if (ok != 0) { Console.WriteLine("fallo en t=" + t + " rc=" + ok); zeroBlocks++; }
            else
            {
                n = Marshal.ReadInt32(list, 0);
                // GetWaveform incluye ambos extremos: para 10 s devuelve 1801.
                // Recortar a la duración exacta evita 1 muestra extra por bloque
                // (60 muestras = 0,33 s de deriva en 10 min).
                int want = (int)Math.Round((blockEnd - t) * SPS / 1000.0);
                int take = Math.Min(n, Math.Max(0, want));
                if (take > 0)
                {
                    IntPtr p = Marshal.ReadIntPtr(list, 4);
                    byte[] raw = new byte[n * REC + 16];
                    Marshal.Copy(p, raw, 0, raw.Length);
                    bool allZero = true;
                    for (int k = 0; k < take; k++)
                    {
                        int o = k * REC;
                        short s0 = BitConverter.ToInt16(raw, o);
                        short s1 = BitConverter.ToInt16(raw, o + 2);
                        short s2 = BitConverter.ToInt16(raw, o + 4);
                        if (s0 != 0 || s1 != 0 || s2 != 0) allZero = false;
                        buffer.Add(s0);
                        buffer.Add(s1);
                        buffer.Add(s2);
                    }
                    totalSamples += take;
                    zeroBlocks = allZero ? zeroBlocks + 1 : 0;
                }
                else zeroBlocks++;
            }
            if (zeroBlocks >= ZERO_BLOCKS && maxSec <= 0) break;
            if (t % 600000 == 0) Console.Write(".");
        }
        Console.WriteLine();

        // Recortar muestras finales en cero (una muestra = 3 short).
        while (buffer.Count >= 3 && buffer[buffer.Count - 1] == 0
               && buffer[buffer.Count - 2] == 0 && buffer[buffer.Count - 3] == 0)
        {
            buffer.RemoveRange(buffer.Count - 3, 3);
            totalSamples--;
        }

        using (var fs = new FileStream(datPath, FileMode.Create, FileAccess.Write))
        using (var bw = new BinaryWriter(fs))
        {
            foreach (short s in buffer) bw.Write(s);
        }
        using (var sw = new StreamWriter(heaPath))
        {
            sw.WriteLine(name + " 3 " + SPS + " " + totalSamples);
            for (int c = 0; c < 3; c++)
                sw.WriteLine(name + ".dat 16 1000.0(0)/mV 16 0 0 0 0 Lead" + (c + 1));
        }
        Console.WriteLine("ventana: inicio=" + startSec.ToString("F3", System.Globalization.CultureInfo.InvariantCulture) + " s, duracion="
                          + (totalSamples / (double)SPS).ToString("F1") + " s");
        Console.WriteLine("exportado " + totalSamples + " muestras (" +
                          (totalSamples / (double)SPS / 3600).ToString("F2") + " h) a " + datPath);
        return;
    }
}
