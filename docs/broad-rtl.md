# BROAD H96: RTL-verifiering

Den nya kärnan matchar heltalsreferensen exakt under **10 000 sammanhängande
H96-steg**: alla 96 tillstånd och fyra råutgångar vid varje steg. RESET
följt av upprepning av det första provet matchar också.

Varje steg tog **124 046 klockcykler** i den kompilerade Verilator-simuleringen.
Vid 50 MHz motsvarar det 2,48092 ms. Detta är kärnans beräkningstid;
USB-tid ingår inte. Hela designen har också klarat
[placering/routing](broad-build.md) med Fmax 50,5561 MHz mot kravet 50 MHz.
[15 000 fysiska steg](../artifacts/broad-h96-deployment-v2/physical/report.json)
matchar exakt, varav 1 000 rörelseprov; kortets klocka är 50 MHz.

Kärnan använder en delad signerad 16×16-multiplikator, 48-bitars ackumulator,
fyra lokala mellanresultat och separata gamla/nya tillstånd. Vikterna ligger
i 41 synkront lästa banker om 1 024 ord. Aktiveringstabellerna läses också
synkront. Indata och utgångar följer det låsta Q11/Q13-formatet.
I de packade RTL-portarna ligger kanal/tillstånd noll i minst signifikanta
ordet. Textvektorer listar däremot indata noll först, från vänster.

Ett separat Icarus-test kör en verklig H3-export genom tre återkopplade steg,
RESET och ett upprepat första steg. Det kontrollerar även okända värden,
busy/done och cykelräknaren mot observerade klockslag. H3 tog 542 cykler.
Den senaste verifieringen omfattar 94 unika tester i två separata körningar:
kapacitetstestet och övriga 93 tester. Compileall, strikt Verilator-lint för
H3/H96 och oberoende granskning passerade. [V4-rapporten](../artifacts/broad-rtl-v4/report.json)
binder den slutliga kärnan till 10 000 exakta steg och RESET-replay.
Explicita tillståndsskrivningar och tidigare registrering av `gate_sum`
ändrade varken resultaten eller de 124 046 cyklerna per steg. Den nya
kvot/rest-avrundningen och carry-save-additionen bevarar samma heltalsmatematik.

## Reproducera

[Artefakterna](../artifacts/broad-rtl-v4/) innehåller simuleringsrapport,
bankmanifest, testprogram och körskript. Alla 41 viktbanker rekonstruerades
till exakt de 41 092 exporterade koefficienterna; utfyllnaden är noll.
Det publicerade staging-skriptet återskapade alla 45 staging-filer byte för byte.

Kör först [heltalsstudien](broad-integer.md). Följande kräver nya utmappar,
Verilator och en C++-kompilator. V4-körskriptet binder den publicerade
V3-rapporten för att kontrollera oförändrat cykelantal:

```sh
.venv/bin/python artifacts/broad-rtl-v4/stage-replay.py --export-dir build/broad-integer-v1/export --vectors-file build/broad-integer-v1/validation/validation/02.hex --output-dir build/broad-rtl-v4
cp artifacts/broad-rtl-v4/replay.cpp build/broad-rtl-v4/replay.cpp
cp artifacts/broad-rtl-v4/run-replay.py build/broad-rtl-v4/run-replay.py
.venv/bin/python build/broad-rtl-v4/run-replay.py
```

Den ursprungliga skalära kärnan är oförändrad. [H96:s sluttest och fysiska
USB-verifiering](broad-h96-results.md) redovisas separat. Den arkiverade
V3-implementationen avvek på kortet; V4 och den nya driftsättningen bevarar
modellen och testresultaten utan att skriva om det ursprungliga studieunderlaget.
Den äldre skalära designens 600-sekunders USB-uthållighetskrav är fortfarande öppet.
