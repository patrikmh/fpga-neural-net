# BROAD H96: RTL-verifiering

Den nya kärnan matchar heltalsreferensen exakt under **10 000 sammanhängande
H96-steg**: alla 96 tillstånd och fyra råutgångar vid varje steg. RESET
följt av upprepning av det första provet matchar också.

Varje steg tog **124 046 klockcykler** i den kompilerade Verilator-simuleringen.
Vid 50 MHz motsvarar det 2,48092 ms. Detta är kärnans beräkningstid;
USB-tid, faktisk FPGA-timing och fysisk verifiering återstår.

Kärnan använder en delad signerad 16×16-multiplikator, 48-bitars ackumulator,
fyra lokala mellanresultat och separata gamla/nya tillstånd. Vikterna ligger
i 41 synkront lästa banker om 1 024 ord. Aktiveringstabellerna läses också
synkront. Indata och utgångar följer det låsta Q11/Q13-formatet.
I de packade RTL-portarna ligger kanal/tillstånd noll i minst signifikanta
ordet. Textvektorer listar däremot indata noll först, från vänster.

Ett separat Icarus-test kör en verklig H3-export genom tre återkopplade steg,
RESET och ett upprepat första steg. Det kontrollerar även okända värden,
busy/done och cykelräknaren mot observerade klockslag. H3 tog 542 cykler.
Alla 86 tester och compileall passerade; strikt Verilator-lint passerade
för H3 och H96. Oberoende granskning verifierade RTL, viktbanker och
hashkedjan till det större simuleringsresultatet.

## Reproducera

[Artefakterna](../artifacts/broad-rtl-v1/) innehåller simuleringsrapport,
bankmanifest, testprogram och körskript. Alla 41 viktbanker rekonstruerades
till exakt de 41 092 exporterade koefficienterna; utfyllnaden är noll.
Det publicerade staging-skriptet återskapade alla 45 staging-filer byte för byte.

Kör först [heltalsstudien](broad-integer.md). Följande kräver nya utmappar,
Verilator och en C++-kompilator:

```sh
.venv/bin/python artifacts/broad-rtl-v1/stage-replay.py --export-dir build/broad-integer-v1/export --vectors-file build/broad-integer-v1/validation/validation/02.hex --output-dir build/broad-rtl-v1
cp artifacts/broad-rtl-v1/replay.cpp build/broad-rtl-v1/replay.cpp
.venv/bin/python artifacts/broad-rtl-v1/run-replay.py
```

Den ursprungliga skalära kärnan är oförändrad. En preliminär syntes av enbart
H96-kärnan använder 48 blockminnen, men är inget bevis för att hela designen
ryms och klarar timing. UART-kedjan, fullständig placering/routing och
fysisk USB-verifiering är nästa steg. Nya sluttestdata är fortfarande oöppnade.
