# ACSE-4-armageddon

Asteroids entering Earth’s atmosphere are subject to extreme drag forces that decelerate, heat and disrupt the space rocks. The fate of an asteroid is a complex function of its initial mass, speed, trajectory angle and internal strength. 

[Asteroids](https://en.wikipedia.org/wiki/Asteroid) 10-100 m in diameter can penetrate deep into Earth’s atmosphere and disrupt catastrophically, generating an atmospheric disturbance ([airburst](https://en.wikipedia.org/wiki/Air_burst)) that can cause [damage on the ground](https://www.youtube.com/watch?v=tq02C_3FvFo). Such an event occurred over the city of [Chelyabinsk](https://en.wikipedia.org/wiki/Chelyabinsk_meteor) in Russia, in 2013, releasing energy equivalent to about 520 [kilotons of TNT](https://en.wikipedia.org/wiki/TNT_equivalent) (1 kt TNT is equivalent to 4.184e12 J), and injuring thousands of people ([Popova et al., 2013](http://doi.org/10.1126/science.1242642); [Brown et al., 2013](http://doi.org/10.1038/nature12741)). An even larger event occurred over [Tunguska](https://en.wikipedia.org/wiki/Tunguska_event), an unpopulated area in Siberia, in 1908. 

This tool predicts the fate of asteroids entering Earth’s atmosphere for the purposes of hazard assessment.

### Installation Guide

The tool uses Python and the following Python libraries: 
```
python -m pip install -r requirements.txt
```

To import the files (when in the chosen directory) go onto the terminal and enter:

```
git clone git@github.com:acse-2019/acse-4-armageddon-astraea.git
```

### User instructions

In Python, enter the directory containing acse-4-armageddon-astraea and run the code:

```
import armageddon
```
By doing so, functions from the different files can be called. For example, the solve_atmospheric_entry function can be called like this:

```
x = Planet()

results_dataframe = x.solve_atmospheric_entry(25, 2.0e4, 3000, 1e6, 45) 

```
In the example, the inputs in the function are the radius, velocity, density, strength, and angle of the asteroid at t=0 (ie. the initial state). results_dataframe is a Pandas dataframe with velocity, mass, angle, altitude, distance, radius, and time as the columns, and the corresponding values found by the solver up until airburst, cratering, or both, occurs.

### Documentation

The code includes [Sphinx](https://www.sphinx-doc.org) documentation. On systems with Sphinx installed, this can be built by running

```
python -m sphinx docs html
```

then viewing the `index.html` file in the `html` directory in your browser.

For systems with [LaTeX](https://www.latex-project.org/get/) installed, a manual pdf can be generated by running

```
python -m sphinx  -b latex docs latex
```

Then following the instructions to process the `Armageddon.tex` file in the `latex` directory in your browser.

### Testing

The tool includes several tests, which you can use to checki its operation on your system. With [pytest](https://doc.pytest.org/en/latest) installed, these can be run with

```
python -m pytest armageddon
```
