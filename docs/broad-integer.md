# BROAD H96: heltalskontroll

Den valda modellen klarar den förregistrerade precisionsgränsen på båda
hela valideringsinspelningarna, med tillståndet bevarat mellan tidssteg.

| Mått | Flyttal | Heltal |
|---|---:|---:|
| Full orienterings-RMSE | 26,433936° | 26,436302° |
| Lutnings-RMSE | 11,827967° | 11,826992° |
| Heading-RMSE | 23,794987° | 23,796876° |

Fullständigt orienteringsfel ökade med **0,002365°**, under den låsta gränsen
0,25°. Inga indata, tillstånd eller råutgångar klipptes och inga råa
nollquaternioner uppstod. Aktiveringstabellernas index begränsades till
tabellområdet 13 085 gånger; dessa händelser redovisas separat och ingår i
den uppmätta noggrannheten.

Indata använder signerade 16-bitars Q11-ord; vikter, tillstånd och fyra
råutgångar använder Q13. Ackumulatorn är signerad med 48 bitar och 26
bråkbitar. Avrundning sker till närmaste, med halvor bort från noll.
Alla neuroner läser föregående tillstånd; utgångslagret läser det nya.
De fyra råutgångarna normeras till en quaternion på datorn.

## Verifiering och reproduktion

[Planen](../experiments/broad-numeric-study-v1.json) låstes före utvärderingen.
[Artefakterna](../artifacts/broad-integer-v1/) innehåller självbärande export,
rapport, oberoende kontroll, källhashar och den använda köraren.
En separat skalär beräkning matchade samtliga 96 tillstånd och fyra
råutgångar vid 160 kontrollerade H96-tidssteg, inklusive steg inne i
inspelningarna. De sparade prediktionernas orienteringsmått räknades om
oberoende av utvärderaren.

```sh
.venv/bin/python artifacts/broad-integer-v1/run-study.py
```

Kör från projektroten med samma förberedda tränings-/valideringsmanifest.
Utmapparna måste vara nya. Köraren använder endast träning och validering;
sluttestinspelningarna 04/08/12/31 är fortfarande oöppnade.

82 tester och compileall passerade efter implementationen. Därefter
passerade ytterligare tre gränsfall tillsammans med integrationstestet:
avrundning/klippning, felaktiga artefakter och nollutgångar även utanför
poängsatta tidssteg. Produktionskällorna ändrades inte däremellan.

Detta är CPU-verifiering av heltalsmodellen. RTL-verifiering, faktisk
syntes/routing och fysisk körning av H96 återstår.
