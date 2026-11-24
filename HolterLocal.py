"""Punto de entrada de la aplicación (también del .exe)."""
import sys

if __name__ == '__main__':
    if len(sys.argv) > 2 and sys.argv[1] == '--motor-holterai':
        from holterlocal.motor_holterai import main as trabajador
        sys.exit(trabajador(sys.argv[2]))
    from holterlocal.app import main
    sys.exit(main())
