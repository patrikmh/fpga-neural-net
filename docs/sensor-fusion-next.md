# Nästa steg: verkliga sensordata

Rekommenderad kandidat är [BROAD](https://github.com/dlaidig/broad):
accelerometer, gyroskop och magnetometer med synkroniserat optiskt
orienteringsfacit. Datasetet har 39 inspelningar och licens CC BY 4.0.
Källversionen är låst till `7e2f8189bb4a3ac91eae32c69b6bc87f08a6e2ad`.

## Första faktiska försöket

Tre inspelningar är hämtade: 01 långsam rotation, 06 snabb rotation och
28 kombinerad rörelse nära en stationär magnet. Totalt 12 774 837 byte
sensordata. De har nio kanaler, 285,714 Hz och orientering som quaternion
`(w,x,y,z)` i ENU. Det är ett utforskande urval, inte ett låst testset.

En enkel kausal accelerometer/gyro-baslinje kördes på datorn. Den integrerar
gyro med Rodrigues rotation och korrigerar upp-riktningen med accelerometern,
med tidskonstant 0,5 s bestämd före utvärderingen. Den initieras från första
accelerometervärdet, aldrig från det optiska facit. Alla tidssteg behålls;
felpoäng räknas på rörelsefas med giltigt facit.

| Inspelning | Bara accelerometer | Bara gyro, init från accel | Accel + gyro |
|---|---:|---:|---:|
| 01 långsam rotation | 5,65° | 17,27° | **0,78°** |
| 06 snabb rotation | 14,62° | 3,08° | **2,44°** |
| 28 magnet/rotation/translation | 37,20° | 12,87° | **9,60°** |

Måttet är lutnings-RMSE i grader, inte heading eller fullständigt
orienteringsfel. Baslinjen använder inte magnetometern. Vid detta förhandsförsök tränades ingen CfC-modell. Det efterföljande
[CfC-försöket och dess slutresultat](broad-results.md) finns nu dokumenterade;
sensorfusion har ännu inte laddats på FPGA:n.
Fullständiga mätningar och versioner finns i
[försöksrapporten](../reports/broad-preview-v1-2026-10-20.json).

## Två möjliga fortsättningar

**BROAD: skatta 3D-orientering (rekommenderas).** Ger ett tydligt facit och
möjlighet att mäta vinkel-, heading- och lutningsfel, även under störningar.
Det kräver flera sensorindata och en ny utgångsrepresentation; nuvarande
FPGA-kärna och protokoll har en enda indata och en enda utgång. Datasetets
samplingsfrekvens är också högre än den nu uppmätta USB-genomströmningen.
Börja därför med en separat flyttalsmodell på datorn och planera hårdvaran
först efter att kvalitet och datatakt är mätta.

**UCI HAR: känna igen rörelse.**
[Detta dataset](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)
har accelerometer och gyro vid 50 Hz. Det är ett enklare första demonstrationsmål,
men klassificerar aktiviteter i stället för att skatta fysisk orientering.
Även det kräver ny modell och flerkanalsindata.

Patrik valde BROAD-orientering. Det första CPU-försöket är klart enligt
[resultatrapporten](broad-results.md). Därefter beställdes en större modell,
med noggrannhet som högsta prioritet och faktisk FPGA-kapacitet som gräns.
Den verifierade skalära modellen behålls som fungerande referens.
För en framtida BROAD-modell måste normalisering tränas enbart på träningsdata,
hela inspelningar hållas isär mellan träning/validering/test och facitluckor
maskeras utan att tidslinjen kollapsas. Publicerade algoritmparametrar valda
över alla 39 inspelningar får inte användas som en oberoende testbaslinje.

## Reproducera förhandsförsöket

Analysen är separerad från produktens Pythonmiljö och ändrar inte `uv.lock`:

```sh
uv venv build/broad-analysis-env --python .venv/bin/python
uv pip install --python build/broad-analysis-env/bin/python h5py==3.16.0 numpy==2.4.6
```

[Nedladdningsmanifestet](../experiments/broad-download-manifest.json) innehåller
de exakta källadresserna och SHA-256. Lägg filerna under
`build/datasets/broad-preview/` med manifestets relativa sökvägar och kopiera
manifestet till `download-manifest.json` i samma katalog. Då kan
[analyskoden](../experiments/broad_tilt_preview.py) köras:

```sh
build/broad-analysis-env/bin/python experiments/broad_tilt_preview.py --output build/broad-preview/result.json
```

Utfilen måste vara ny. Inga datasetfiler ingår i Git-repot.

Källa: D. Laidig, M. Caruso, A. Cereatti och T. Seel (2021),
[BROAD—A Benchmark for Robust Inertial Orientation Estimation](https://doi.org/10.3390/data6070072).
Data används oförändrade under CC BY 4.0; baslinjeanalysen ovan är vår egen.
