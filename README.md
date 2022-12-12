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

  <h3 align="center">CDISCO: Concept Discovery in Deep Spaces with Singular Vector Decomposition</h3>

  <p align="center">
    Repo with the initial toolkit and the basic functionalities to perform concept discovery in the latent space of deep learning models.
    <br />
    <a href="https://github.com/othneildrew/Best-README-Template"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/othneildrew/Best-README-Template">View Demo</a>
    ·
    <a href="https://github.com/othneildrew/Best-README-Template/issues">Report Bug</a>
    ·
    <a href="https://github.com/othneildrew/Best-README-Template/issues">Request Feature</a>
  </p>
</p>



<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Supported Models</summary>
  <ol>
    <li>
      <a href="#about-the-project">State-of-the-art CNNs pretrained on ImageNet</a>
      <ul>
        <li>Inception V3</li>
      </ul>
      <ul>
        <li>ResNet 50</li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">(Hopefully) DistilBERT (a baby-transformer)</a>
    </li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About CDISCO

This repo contains the implementation of the CDISCO toolkit proposed in the paper "nanan". 
The central question of this work is: Given a representation of a complex model such as its deep latent space, is this already an interpretable version? If not, is an interpretable representation of deep spaces a compressed representation of the original space?

### Functionalities

CDISCO has multiple functionalities, which are described in the following. 

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
  
  *

### Built With

This section should list any major frameworks that you built your project using. Leave any add-ons/plugins for the acknowledgements section. Here are a few examples.
* [Python3.6](https://getbootstrap.com)
* [PyTorch](https://jquery.com)
* [IDK](https://laravel.com)


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



<!-- ROADMAP -->
## Roadmap

See the [open issues](https://github.com/othneildrew/Best-README-Template/issues) for a list of proposed features (and known issues).



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

Your Name - [@your_twitter](https://twitter.com/your_username) - email@example.com

Project Link: [https://github.com/your_username/repo_name](https://github.com/your_username/repo_name)



<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
* [GitHub Emoji Cheat Sheet](https://www.webpagefx.com/tools/emoji-cheat-sheet)
* [Img Shields](https://shields.io)
* [Choose an Open Source License](https://choosealicense.com)
* [GitHub Pages](https://pages.github.com)
* [Animate.css](https://daneden.github.io/animate.css)
* [Loaders.css](https://connoratherton.com/loaders)
* [Slick Carousel](https://kenwheeler.github.io/slick)
* [Smooth Scroll](https://github.com/cferdinandi/smooth-scroll)
* [Sticky Kit](http://leafo.net/sticky-kit)
* [JVectorMap](http://jvectormap.com)
* [Font Awesome](https://fontawesome.com)





<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/othneildrew
[product-screenshot]: images/screenshot.png
