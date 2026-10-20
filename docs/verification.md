# Verifierad kedja

Modellträning, låst heltalsreferens, RTL-simulering, syntes, timingkontroll,
SRAM-laddning och fysisk USB-inferens är verifierade. RESET följs av en
sammanhängande ström; varje mottaget utvärde och alla åtta tillstånd jämförs.

Den samlade programkontrollen `make check` passerade 62 tester och compileall.
Den innehåller verklig Icarus-simulering, 10 000 kontinuerliga kärnsteg,
UART-protokolltester, verkligt FPGA-bygge, modellidentiteter och en kompilerad
extern FTDI-testmotpart. Tester programmerar aldrig ansluten hårdvara.

Den fysiska första neurala körningen gav 10 000 av 10 000 exakta svar,
21,04 µs beräkning och 5,89 ms genomsnittlig USB-tid. Den aktuella
[resultatrapporten](../reports/cfc-summary-2026-10-20.json) sammanfattar bygget
och det efterföljande långprovet. Maskinspecifika fullständiga rapporter ligger
lokalt under `reports/`; reproducerbara modellfiler finns under `artifacts/`.
**600-sekunderskravet är fortfarande öppet.** Längre försök stoppades av
USB-läsfel. Det senaste avbrottet skedde även med vilolägesskydd aktivt;
anslutningens långtidsstabilitet behöver undersökas vidare. Avbrutna försök
räknas inte som godkända.

Appen är dessutom prövad i riktig Chrome: livedata, sparad körning, reglage,
paus medan mätningen fortsätter, JSON-nedladdning och mobilvy utan horisontell
rullning. Ett oberoende smoke-test bekräftade graf, åtta tillstånd, växling
mellan lägen, paus och 390-pixelsvy utan konsolfel.

Grafen visar tillstånd efter varje färdigt beräkningssteg. Den är inte en
spårning av enskilda FPGA-klockcykler. Beräkningstiden räknas från den mottagna
cykelräknaren och 50 MHz; USB-tiden mäts separat på datorn. Sparad körning är
märkt och använder inga påhittade individuella USB-tider.

Endast SRAM programmeras. Strömavbrott tar bort konfigurationen. Flash har
inte skrivits. Timingresultatet gäller den sparade bitströmmen och
verktygsuppsättningen; andra byggen måste klara samma kontroll före laddning.
