# BROAD H96: sluttest och hårdvarustatus

H96-modellen med 41 092 parametrar valdes på valideringsdata. Modellen,
Q11/Q13-formatet, utvärderingsreceptet och den byggda bitströmmen låstes
innan fyra nya inspelningar hämtades. [Urvalet](../artifacts/broad-h96-final-v1/selection.json)
frystes 2026-10-20 19:49:48 UTC; nedladdningen började 19:50:01 UTC.
Sluttestet användes inte för modellval eller omträning.

## Noggrannhet på nya inspelningar

RMSE i grader, lägre är bättre. Varje hel inspelning körs sammanhängande
med nollställt tillstånd vid start. Rörelsemarkering och giltig referens
styr poängsättningen; de första 32 proven undantas. Medelvärdet väger de
fyra inspelningarna lika. Totalt poängsattes 125 674 prov.

| Inspelning | Full orientering, heltal | Lutning, heltal | Riktning, heltal |
|---|---:|---:|---:|
| 04, långsam rotation | 18,596° | 8,072° | 16,778° |
| 08, snabb rotation | 40,999° | 16,173° | 38,059° |
| 12, translation | 6,110° | 3,581° | 4,952° |
| 31, magnetstörning | 32,850° | 13,911° | 30,002° |
| Medel | **24,639°** | **10,434°** | **22,448°** |

Flyttalsmodellen ger 24,63803° fullständigt orienteringsfel. Heltalsmodellens
ökning är **0,000805°**. Indata-, tillstånds- och utgångsklippning samt
nollkvaternioner är samtliga noll; 15 776 tabellindex klamrades till LUT-domänen.
144 oberoende skalära H96-steg matchade exakt. En separat kontroll räknade
om fullständigt RMSE från sparade prediktioner och verifierade maskeringen.

Det kausala gyro/accelerometer-filtret ger **4,196° lutningsfel** på samma
inspelningar, bättre än CfC-modellens 10,434°. Modellen har därmed fortfarande
ett tydligt noggrannhetsproblem. Rapportens baslinjejämförelse gäller bara
lutning; jämför därför inte baslinjens lutningsfel med modellens fullständiga orienteringsfel.

[CPU-rapport](../artifacts/broad-h96-final-v1/report.json),
[flyttalsrapport](../artifacts/broad-h96-final-v1/float-report.json) och
[rådatahashar](../artifacts/broad-h96-final-v1/download-manifest.json) är publicerade.
Fullständiga prediktions-NPZ-filer ligger lokalt i `build/broad-h96-final-v1/`;
det publicerade körreceptet återskapar dem.

## Bygg- och funktionskontroller

Hela designen ryms: **11 361 LUT4, 4 218 register och 48 av 56 blockminnen**.
Routingverktyget rapporterar **50,556 MHz Fmax** mot kravet 50 MHz.
Kortets klocka är 50 MHz. [Byggrapporten](../artifacts/broad-h96-deployment-v2/build/report.json)
binder RTL, export, verktyg och bitström med hashar.

[V4-simuleringen](../artifacts/broad-rtl-v4/report.json) matchar alla 96
tillstånd och fyra råutgångar under 10 000 steg, inklusive RESET/replay.
Varje steg tar 124 046 cykler, motsvarande 2,48092 ms vid 50 MHz.
[94 unika tester](../artifacts/broad-h96-deployment-v2/verification.json)
passerade i separata kapacitets- och regressionskörningar. Seed 3 återskapade
exakt samma bitström som den fysiskt verifierade; byggstandardens seed 1
är oförändrad. Byggkommandot måste uttryckligen välja seed 3.

## Fysisk verifiering passerar med den nya implementationen

[Den fysiska rapporten](../artifacts/broad-h96-deployment-v2/physical/report.json)
verifierar **15 000 steg och sex RESET** på det serienummerstyrda kortet.
Alla 96 tillstånd, fyra råutgångar, CRC, sekvens och 124 046 cykler per steg
matchar heltalsreferensen. Ett separat slutprov genom samma publika CLI
verifierade ytterligare fyra rörelseprov efter RESET.

- 10 000 sammanhängande prov från validering 02, utan rörelsemarkering.
- 1 000 rörelsemarkerade valideringsprov från index 11 449, med nytt nolltillstånd.
- Fyra separata RESET + 1 000 prov från sluttest 04/08/12/31, samtliga stationära prefix.

Rörelsefönstrets nollställning skiljer sig från tillståndshistoriken i en hel
inspelning. Resultatet visar bitexakt numerik för dessa prov; det är ingen
fysisk mätning av hela sluttestets orienteringsfel. Modellen, vikterna,
normaliseringen och precisionen är oförändrade. Det ursprungliga urvalet och
CPU-rapporten bevaras; [driftsättningen](../artifacts/broad-h96-deployment-v2/)
binder dem till den nya implementationen utan ett nytt modellval.

## Tidigare fysiskt fel och kvarvarande gränser

Första laddningsförsöket stoppades före programmering av ett USB-fel vid
läsning av FTDI-serienumret. En senare serienummerstyrd JTAG-kontroll
identifierade rätt GW5A-25-kort, och andra försöket laddade bitströmmen i SRAM.
Flash skrevs inte.

Det tidigare bygget klarade RESET, sekvens, CRC och cykeltal men avvek redan
i första beräkningssteget. Gamla valideringsvektorer visade återkommande fel
i tillståndsbit 9. Både RTL- och mappade simuleringar matchade referensen;
den exakta orsaken till skillnaden på kortet är inte fastställd.

Avrundningen skrevs om med kvot/rest och blend-additionen med carry-save.
De nya DUT-kontrollerna och en oberoende matematisk granskning bekräftar
samma heltalsresultat, inklusive negativa halvor och giltiga aktiveringsgränser.
Inga klockcykler tillkom. Det nya bygget passerar ovanstående fysiska prov.

[Försök 1](../artifacts/broad-h96-final-v1/physical-attempt-v1/result.json) och
[försök 2](../artifacts/broad-h96-final-v1/physical-attempt-v2/result.json)
bevarar felen; [diagnostiken](../artifacts/broad-gate-diagnostics-v1/) och
[byggförsöken](../artifacts/broad-h96-deployment-v2/history.json) är också sparade.
Den tidigare skalära modellens 600-sekunders USB-prov är
fortfarande öppet. Appens befintliga vy visar den skalära åttaneuronsmodellen.
