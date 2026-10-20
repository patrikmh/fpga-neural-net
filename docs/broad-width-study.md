# BROAD: större CfC med noggrannhet först

Den förregistrerade CPU-studien är genomförd: fyra bredder med 50 epoker och
seed 7, följt av två 200-epokskörningar för vinnande bredd, seeds 7 och 17.
Totalt sex körningar och 600 epoker. Samma tränings-/valideringsinspelningar,
normalisering, urvalsmått och övriga inställningar användes genomgående.

| Neuroner | Parametrar | Validering, full orienterings-RMSE |
|---|---:|---:|
| 8, tidigare referens | 612 | 34,71° |
| 32 | 5 508 | 29,73° |
| 64 | 19 204 | 28,92° |
| **96** | **41 092** | **26,43°** |
| 104 | 47 844 | 26,48° |

96 neuroner vann enligt det låsta urvalsmåttet. Det ger **23,8 % lägre
valideringsfel** än åtta neuroner. Lutnings-RMSE är 11,83° och heading-RMSE
23,79°. Dessa är valideringsresultat, inte ett nytt oberoende sluttest.

Längre träning förbättrade inte den valda checkpointen: seed 7 hade sitt
bästa resultat vid epok 50 även efter 200 epoker. Seed 17 nådde 29,07°.
Bästa checkpointen från de sex körningarna är sparad i
[artefaktkatalogen](../artifacts/broad-width-study-v1/), tillsammans med
studierapport, källhashar och urvalsbeslut.

## Vad som återstår

**Modellen har ännu inte byggts eller laddats på FPGA:n.** Tang Primer 25K
har 23 040 LUT4, 23 040 register och 56 BSRAM enligt det verifierade verktygsbygget.
För H96 uppskattas vikter till 41 block och aktiveringstabeller till sju block.
48 av 56 block är en uppskattning för det föreslagna lagringssättet; faktisk
syntes, placering, routing och timing krävs innan modellen sägs rymmas.

Den ursprungliga kärnan kan inte bara ändra en breddkonstant: den har skalär
indata/utdata och lagrar alla neuroners mellanresultat i register. För större
nät planeras en delad MAC och fyra lokala 48-bitars mellanresultat per neuron,
med gammalt tillstånd bevarat tills hela tidssteget är färdigt. CPU-modellens
quaternionnormalisering måste också hanteras uttryckligen i hårdvarukedjan.

Indataformatet behöver ändras: Q13 skulle klippa 9 212 träningsvärden och
12 668 valideringsvärden. Q10 täcker båda utan klippning. Det är en kandidat,
inte ett färdigt bevis för kvantiserad noggrannhet.

Träningen återställer minnet efter 256 prov, cirka 0,896 s, medan valideringen
kör hela inspelningar på cirka 175–186 s. Detta är en konstaterad skillnad i
upplägg; dess betydelse för felet är ännu en hypotes. Det godkända
[försöket med sammanhängande träning](broad-stateful-study.md) är nu genomfört.
Ingen av två körningar förbättrade startmodellen. Samma H96-checkpoint går
därför vidare till heltalskontroll och FPGA-bygge.

Nya sluttestinspelningar **04, 08, 12 och 31 är inte öppnade**. De tidigare
03/07/11/29 får bara kallas återanvända regressionstest. Inga testinspelningar
användes för att välja bredd eller checkpoint i denna studie.

## Reproducera CPU-studien

Använd källversion `721b44a`, projektets låsta `.venv` och det publicerade
[förregistrerade upplägget](../experiments/broad-width-study-v1.json).
Träningsdata återskapas enligt [första försöket](broad-results.md).
Den ursprungliga konverteraren ger exakt samma träningsmanifest.

```sh
for H in 32 64 96 104; do
  .venv/bin/python -m model.broad_train --prepared-dir build/broad-train-prepared --output-dir "build/screen-h${H}-seed7" --hidden-size "$H" --epochs 50 --seed 7
done
for SEED in 7 17; do
  .venv/bin/python -m model.broad_train --prepared-dir build/broad-train-prepared --output-dir "build/refine-h96-seed${SEED}" --hidden-size 96 --epochs 200 --seed "$SEED"
done
```

Alla utmappar måste vara nya eller tomma. Välj enbart på medelvärdet av
inspelningarnas fullständiga orienterings-RMSE i valideringsrapporten.

76 tester och compileall passerade efter breddändringen. Granskningen
bekräftade H8-kompatibilitet och H32/H64-former. Den parameteriserade koden
återgav dessutom referensmodellens sparade H8-valideringsprediktioner exakt.
Oberoende CLI-smoke med den sparade H96-checkpointen passerade på syntetiska
HDF5-data: 64 ändliga normerade quaternioner, exakt 31 poängsatta prov efter
maskering och identiska prediktioner vid separat omladdning. Ingen riktig
sluttestdata eller FPGA berördes.
Den äldre USB-stabilitetsgrinden på 600 sekunder är fortfarande separat och öppen.
