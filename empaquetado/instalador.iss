; Inno Setup: instalador de HolterLocal para Windows (sin permisos de administrador).
#define Version GetEnv("HL_VERSION")
[Setup]
AppName=HolterLocal
AppVersion={#Version}
AppPublisher=HolterLocal
DefaultDirName={localappdata}\Programs\HolterLocal
DefaultGroupName=HolterLocal
PrivilegesRequired=lowest
OutputDir=..\salida
OutputBaseFilename=HolterLocal-{#Version}-instalador
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayIcon={app}\HolterLocal.exe
[Languages]
Name: "es"; MessagesFile: "compiler:Languages\Spanish.isl"
[Tasks]
Name: "escritorio"; Description: "Crear un acceso directo en el escritorio"; GroupDescription: "Accesos directos:"
[Files]
Source: "..\dist\HolterLocal\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion
[Icons]
Name: "{group}\HolterLocal"; Filename: "{app}\HolterLocal.exe"
Name: "{group}\Desinstalar HolterLocal"; Filename: "{uninstallexe}"
Name: "{userdesktop}\HolterLocal"; Filename: "{app}\HolterLocal.exe"; Tasks: escritorio
[Run]
Filename: "{app}\HolterLocal.exe"; Description: "Abrir HolterLocal"; Flags: nowait postinstall skipifsilent
