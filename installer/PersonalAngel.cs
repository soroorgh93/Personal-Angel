// PersonalAngel.exe — tiny native Windows launcher (no console window).
// Compiled by scripts\setup_windows.ps1 with the C# compiler that ships with every Windows
// installation (.NET Framework 4.x: C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe),
// so no extra download is needed. It starts the app in desktop mode from the project's
// virtual environment:  .venv-win\Scripts\pythonw.exe -m personal_angel desktop --profile <p>
//
// Usage:  PersonalAngel.exe            (profile from profile.txt, default pc_cpu)
//         PersonalAngel.exe pc_gpu     (explicit profile)
//         PersonalAngel.exe fixture    (rehearsal mode, no models needed)
using System;
using System.Diagnostics;
using System.IO;
using System.Windows.Forms;

static class Launcher
{
    [STAThread]
    static void Main(string[] args)
    {
        string root = AppDomain.CurrentDomain.BaseDirectory;
        string pyw = Path.Combine(root, ".venv-win", "Scripts", "pythonw.exe");
        string py = Path.Combine(root, ".venv-win", "Scripts", "python.exe");
        string exe = File.Exists(pyw) ? pyw : py;
        if (!File.Exists(exe))
        {
            MessageBox.Show("PersonalAngel is not set up yet.\n\nOpen PowerShell in this folder and run:\n    .\\scripts\\setup_windows.ps1",
                            "PersonalAngel", MessageBoxButtons.OK, MessageBoxIcon.Warning);
            return;
        }
        string profile = "pc_cpu";
        string profileFile = Path.Combine(root, "profile.txt");
        if (File.Exists(profileFile))
        {
            string p = File.ReadAllText(profileFile).Trim();
            if (p.Length > 0) profile = p;
        }
        if (args.Length > 0 && args[0].Length > 0) profile = args[0];
        var psi = new ProcessStartInfo(exe, "-m personal_angel desktop --profile " + profile)
        {
            WorkingDirectory = root,
            UseShellExecute = false,
            CreateNoWindow = true,
        };
        try
        {
            Process.Start(psi);
        }
        catch (Exception ex)
        {
            MessageBox.Show("Could not start PersonalAngel:\n" + ex.Message + "\n\nSee runs\\desktop.log", "PersonalAngel",
                            MessageBoxButtons.OK, MessageBoxIcon.Error);
        }
    }
}
