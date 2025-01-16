Set WshShell = CreateObject("WScript.Shell")
CurrentDirectory = CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName)
WshShell.CurrentDirectory = CurrentDirectory

' Önceki süreçleri temizle
Set oShell = CreateObject("WScript.Shell")
oShell.Run "taskkill /F /IM pythonw.exe /T", 0, True
WScript.Sleep 1000 ' Süreçlerin kapanmasını bekle

' 8501 portunu kontrol et
Set WshExec = oShell.Exec("netstat -ano | findstr :8501")
strOutput = WshExec.StdOut.ReadAll
If strOutput <> "" Then
    ' Port kullanımda, süreci sonlandır
    Set regex = New RegExp
    regex.Pattern = "LISTENING\s+(\d+)"
    Set matches = regex.Execute(strOutput)
    If matches.Count > 0 Then
        pid = matches(0).SubMatches(0)
        oShell.Run "taskkill /F /PID " & pid, 0, True
    End If
End If

' Streamlit'i başlat (pythonw ile tamamen gizli)
Set oExec = oShell.Exec(CurrentDirectory & "\venv\Scripts\pythonw.exe -m streamlit run app.py --server.headless true")

' Streamlit'in başlamasını bekle
Dim ready : ready = False
Dim attempts : attempts = 0
Do While Not ready And attempts < 30 ' 30 saniye bekle
    WScript.Sleep 1000 ' 1 saniye bekle
    
    ' Port kontrolü
    Set WshExec = oShell.Exec("netstat -ano | findstr :8501")
    strOutput = WshExec.StdOut.ReadAll
    If InStr(strOutput, "LISTENING") > 0 Then
        ready = True
    End If
    
    attempts = attempts + 1
Loop

If ready Then
    ' Tarayıcıyı aç
    WScript.Sleep 2000 ' Son bir bekleme
    oShell.Run "http://localhost:8501", 1, False
    
    ' Streamlit kapanana kadar bekle
    Do While oExec.Status = 0
        WScript.Sleep 1000
    Loop
End If

' Tüm süreçleri temizle
oShell.Run "taskkill /F /IM pythonw.exe /T", 0, True
