# Particle Filter
---
​		Particle filter 2D localization implemented in C++ with Volume.

​		This particle filter includes a **<u>2D LiDAR simulator</u>**, which is based on the **<u>Volume2D Shader of mine</u>**[[Github Repo: Enigmatisms/Volume]](https://github.com/Enigmatisms/Volume). Using this LiDAR simulator, I implemented an interesting little localization program via **<u>Particle Filter</u>**. The localization experiments are only done in a 2D-2DoF (translation position x and y) problem.

​		Under the condition of 2000 particles, the FPS of this algorithm is about 16-50 hz, and the convergence is fast and accurate.

---

### Result

|          ![](./asset/img.png)          |           ![](./asset/img2.png)            |
| :------------------------------------: | :----------------------------------------: |
| Long-corridor problem intial condition |      Long-corridor problem 4-th move       |
|         ![](./asset/img3.png)          |           ![](./asset/ray2.png)            |
|    Long-corridor problem 15-th move    | LiDAR simulator (with noise) visualization |

