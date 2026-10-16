using System;
using System.Reflection;

namespace DatasetProcessor.src.Helpers;

public static class AppVersionHelper
{
    private static string? _version;
    private const string BaseTitle = "Dataset Processor All-in-one Tools";

    public static string Version
    {
        get
        {
            if (_version != null)
            {
                return _version;
            }

            Assembly? entryAssembly = Assembly.GetEntryAssembly();
            Assembly assembly = (entryAssembly != null && (entryAssembly.FullName?.StartsWith("DatasetProcessor") ?? false))
                ? entryAssembly
                : typeof(AppVersionHelper).Assembly;

            string? informationalVersion = assembly.GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion;
            if (!string.IsNullOrWhiteSpace(informationalVersion))
            {
                int plusIndex = informationalVersion.IndexOf('+');
                _version = plusIndex > 0 ? informationalVersion[..plusIndex] : informationalVersion;
                return _version;
            }

            Version? assemblyVersion = assembly.GetName().Version;
            if (assemblyVersion != null)
            {
                _version = assemblyVersion.Build >= 0
                    ? $"{assemblyVersion.Major}.{assemblyVersion.Minor}.{assemblyVersion.Build}"
                    : $"{assemblyVersion.Major}.{assemblyVersion.Minor}";
                return _version;
            }

            _version = "1.0.0";
            return _version;
        }
    }

    public static string Title => $"{BaseTitle} - v{Version}";
}
