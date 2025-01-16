On Error Resume Next

Set WshShell = CreateObject("WScript.Shell")
CurrentDirectory = CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName)
WshShell.CurrentDirectory = CurrentDirectory

' Önceki süreçleri temizle
Set oShell = CreateObject("WScript.Shell")
oShell.Run "taskkill /F /IM pythonw.exe /T", 0, True
oShell.Run "taskkill /F /IM python.exe /T", 0, True
WScript.Sleep 2000

' Hata kontrolü
If Err.Number <> 0 Then
    MsgBox "Süreç temizleme hatası: " & Err.Description, vbExclamation
    WScript.Quit
End If
On Error Goto 0

' Streamlit'i başlat
On Error Resume Next
Set oExec = oShell.Exec("cmd /c " & CurrentDirectory & "\venv\Scripts\activate.bat && " & CurrentDirectory & "\venv\Scripts\streamlit run " & CurrentDirectory & "\app.py")

' Hata kontrolü
If Err.Number <> 0 Then
    MsgBox "Streamlit başlatma hatası: " & Err.Description, vbExclamation
    WScript.Quit
End If
On Error Goto 0

' Streamlit'in başlamasını bekle
WScript.Sleep 5000

' Tarayıcıyı aç
On Error Resume Next
oShell.Run "http://localhost:8501", 1, False
If Err.Number <> 0 Then
    MsgBox "Tarayıcı açma hatası: " & Err.Description, vbExclamation
End If
