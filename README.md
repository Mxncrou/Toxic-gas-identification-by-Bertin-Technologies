# Toxic gas identification by Bertin Technologies

Solution to the Toxic gas identification challenge by Bertin Technologies, hosted on [Challenge Data](challengedata.ens.fr) (ENS Paris, Collège de France, Institut Louis Bachelier).

# The challenge
All informations can be found at [this link](https://challengedata.ens.fr/challenges/156) 

The challenge goal is to predict an alarm level between 0 and 1 for 23 classes (the gas itself and its characteristics, e.g. nerve agent or not). Missing an alarm is penalised more than raising a false one because raising a false one doesn't injure anyone but not raining an alarm when the gas is truly toxic risk lives of people.


For the data, each sample contains:
- 8 sensors from two groups of neighbouring sensors: M4–M7 and M12–M15,
- 4 complementary sensors: S1–S3 and R,
- Humidity, the absolute humidity (because it affects the measurements).

The targets are c1–c23, each between 0 and 1.

**Source:** data provided by Bertin Technologies through Challenge Data. Datasets on Challenge Data are distributed under the Licence Ouverte Etalab unless the challenge page states otherwise.
>The data is not included in this repository. Register on the challenge page, download the files and place them in data/

# Reference

```BibTex
@misc{bertin2026toxicgas,
    title        = {Toxic gas identification},
    author       = {{Bertin Technologies}},
    howpublished = {Challenge Data, ENS Paris, Collège de France and Institut Louis Bachelier},
    year         = {2026},
    url          = {https://challengedata.ens.fr/challenges/156},
    note         = {Accessed: 2026-11-04}
}
```