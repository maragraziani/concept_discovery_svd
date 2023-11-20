<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]



<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/othneildrew/Best-README-Template">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">CDISCO: Concept Discovery in Deep Spaces with Singular Value Decomposition</h3>

  <p align="center">
   Perform concept discovery in the latent space of deep learning models with Singular Value Decomposition. 
    <br />
    <a href="https://github.com/maragraziani/cdisco"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/maragraziani/cdisco">View Demo</a>
    ·
    <a href="https://github.com/maragraziani/cdisco/issues">Report Bug</a>
  </p>
</p>



<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Supported Models</summary>
  <ol>
    <li>
      <a href="#about-the-project">State-of-the-art CNNs</a>
      <ul>
        <li>Inception V3</li>
      </ul>
      <ul>
        <li>ResNet 50</li>
      </ul>
    </li>
  </ol>
</details>
<details open="open">
  <summary>Datasets</summary>
  <ol>
    <li>
      <a href="https://www.image-net.org/">ImageNet</a>
      <ul>
        <li><a href="https://github.com/fastai/imagenette">ImageWoof</a></li>
      </ul>
      <ul>
        <li><a href="https://github.com/fastai/imagenette">Imagenette</a></li>
      </ul>
    </li>
    <li>
      <a href="https://open-xai.github.io/">XAI Benchmark</a>
      <ul>
        <li><a href="https://www.propublica.org/datastore/dataset/compas-recidivism-risk-score-data-and-analysis">COMPAS</a></li>
      </ul>
      <ul>
        <li>...</li>
      </ul>
    </li>
  </ol>
</details>


<!-- ABOUT THE PROJECT -->
## About CDISCO

This repo contains the implementation of the CDISCO toolkit proposed in the paper "Concept Discovery and Dataset Exploration with Singular Value Decomposition". 
The central question of this work is: Given a representation of a complex model such as its deep latent space, is this already an interpretable version? If not, is an interpretable representation of deep spaces a compressed representation of the original space?
We propose to analyze the latent space of a deep neural network with Singular Value Decomposition, to discover a new representation of the space that best describes "what the model has learnt". By reweighting the singular vectors with a gradient-informed ranking, we identify directions in the latent space carrying the most relevant information for the model outcome. 

### Functionalities

CDISCO can be used to identify the singular vectors, to visualize concept maps and to analyze the model internal state. 

CDISCO - main tool
```sh
   import cdisco.cdisco
   ```
* to implement - run_cdisco(model, input_data, save_fold='') -> runs get_model_state() and then discovery()
* discovery(conv_maps, gradients, prediction, classes) -> concept_candidates, eigenvectors
* get_model_state(model, input_data, save_fold='') -> performs inference on input_data and stores the output in save_fold

CDISCO - vis tool

The visualization toolbox allows us to visualize and interpret the results of CDISCO. 
```sh
   import cdisco.vis
   ```
* cdisco.vis.cdisco_concept_vis(image_path, concept_vector, conv_maps) -> concept_heatmap
* cdisco.vis.cdisco_vis_extremes_extensive(concepts_list, concept_candidates, eigenvectors, conv_maps, input_paths, predictions, save_fold='') -> visualizes the top 5 images that have the highest projection on the concept direction and saves it in save_fold
* cdisco.vis.conceptbard(concept, save_fold='') -> saves in save_fold a visualization of the cncept segmentations to create a board that is representative of the concept

CDISCO - analyze tool

```sh
   import cdisco.analyze
   ```
  
  * cdisco.analyze.cdisco_alignment(concepts, concept_candidates) -> prints the list of classes that share the same discovered concept direction 
  * cidsco.analyze.cdisco_pop_concepts(concept_candidates, classes, eigenvectors, save_fold='')-> prints the top 3 most popular directions among the classes
  * cdisco.analyze.cdisco_angle_dissection(eigenvectors, candidates, save_fold='')-> stores in savefold the results of the alignment evaluation of each concept with the canonical basis. 
  
  
### Built With

This section should list any major frameworks that you built your project using. Leave any add-ons/plugins for the acknowledgements section. Here are a few examples.
* [Python3.6](https://getbootstrap.com)
* [PyTorch](https://jquery.com)


<!-- GETTING STARTED -->
## Getting Started

Follow the steps below to install the toolkit as a github repo. Pip install will be implemented soon. 

### Installation

1. Clone this repo
   ```sh
   git clone https://github.com/maragraziani/CDISCO.git
   ```
2. Install required packages
   ```sh
   pip install -r requirements.txt
   ```

<!-- USAGE EXAMPLES -->
## Usage and Functionalities

You can import the repo by running 
   ```python
   import cdisco
   
   ```

_For more examples, please refer to the [Documentation](https://example.com)_



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.


<!-- CONTACT -->
## Contact

Mara Graziani - [@mormontre](https://twitter.com/mormontre) - mara.graziani@hevs.ch


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/maragraziani/cdisco.svg?style=for-the-badge
[contributors-url]: https://github.com/maragraziani/cdisco/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/maragraziani/cdisco.svg?style=for-the-badge
[forks-url]: https://github.com/maragraziani/cdisco/network/members
[stars-shield]: https://img.shields.io/github/stars/maragraziani/cdisco.svg?style=for-the-badge
[stars-url]: https://github.com/maragraziani/cdisco/stargazers
[issues-shield]: https://img.shields.io/github/issues/maragraziani/cdisco?style=for-the-badge
[issues-url]: https://github.com/maragraziani/cdisco/issues
[license-shield]: https://img.shields.io/github/license/maragraziani/cdisco.svg?style=for-the-badge
[license-url]: https://github.com/maragraziani/cdisco/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/mara-graziani-878980105/?originalSubdomain=ch
[product-screenshot]: images/screenshot.png
