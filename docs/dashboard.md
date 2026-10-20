# Se vad som händer på FPGA:n

Appen körs lokalt och visar den senaste verifierarens data. Den öppnar ingen
USB-enhet och programmerar inte kortet. Värdprogrammet är ensamt om USB-porten.

## Öppna appen

```sh
.venv/bin/python -m host.demo --live-dir build/live-session --port 8765
```

Öppna sedan <http://127.0.0.1:8765>. Välj **Sparad körning** för att utforska
värden som matchade exakt i det fysiska 10 000-stegsprovet. Enskilda USB-tider
sparades inte i det provet; appen visar därför dess sammanfattade statistik.
Sparad körning är tydligt märkt och visas aldrig som live.

## Kör live

Kortet behöver köra den verifierade neurala SRAM-bitströmmen. I en annan
terminal, från samma projektkatalog:

```sh
make verify-cfc USB_SERIAL=DITT_USB_SERIENUMMER \
  EXPORT_DIR=artifacts/cfc8-q13-v1 \
  CFC_DURATION=600 CFC_LIVE_DIR=build/live-session \
  CFC_OUTPUT=reports/min-live-korning.json
```

Välj en ny tom livekatalog och ett nytt rapportnamn för varje körning.
Servern och verifieraren måste använda samma livekatalog. Välj **Live** i appen.
En felaktig checksumma, sekvens, beräkning eller timeout avbryter verifieringen
och visar felstatus. Grafen kan pausas utan att mätningen stoppas.

På Mac med nätström: kör långprovet som `caffeinate -disu make verify-cfc ...`
med samma Make-argument. Det håller system och bildskärm vakna tills kommandot
avslutas, utan att ändra datorns inställningar. Tidigare försök avbröts av
systemvila och underhållsvila; enbart `-i` räckte inte i det läget.
Make sätter bibliotekssökvägen efter
`caffeinate`, vilket också undviker att macOS filtrerar bort `DYLD_*`-variabler.
Även en skyddad körning har fått USB-läsfel. Vilolägesskyddet är därför inte
ett bevis på att anslutningen är stabil; det godkända 600-sekundersprovet återstår.

Den blå kurvan visar insignalen. FPGA:ns gröna prognos och den streckade
heltalsreferensen förskjuts ett sampel framåt eftersom nätet förutsäger nästa
värde. Tillståndsstaplarna visar de åtta återkopplade enheterna. Amplituderna
visas i normaliserade enheter. Appen visar de senaste högst 512 stegen.

Beräkningstid kommer från FPGA:ns cykelräknare vid 50 MHz. USB-tiden mäts
separat på datorn. Ett godkänt långprov kräver minst 600 sekunder och minst
100 verifierade sampel/s utan fel. Endast en initial RESET används; tillståndet
fortsätter även när referensinsignalen börjar om.

## Bygg och ladda SRAM

```sh
make cfc-build EXPORT_DIR=artifacts/cfc8-q13-v1 CFC_BUILD_DIR=build/cfc-session
```

Byggkatalogen måste vara ny eller tom. Rapporten innehåller verktyg, käll- och
modellhashar, minnesmappning och timing. Kontrollera att rapporten är godkänd
före laddning. Med den lokala verktygsinstallationen på Mac:

```sh
.tools/tang-toolchain/oss-cad-suite/bin/openFPGALoader \
  -b tangprimer25k --usb-serial-num DITT_USB_SERIENUMMER \
  --write-sram build/cfc-session/cfc.fs
```

Använd aldrig flashflaggor. Efter strömavbrott behöver SRAM laddas på nytt.
Programmeraren och verifieraren ska köras i turordning.
