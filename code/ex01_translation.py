from utils import images
import numpy as np


def main():
    # Pre transformacije
    img = images.load_image('doge.png')
    img = images.add_axis(img)
    fig = images.plot_image(img)
    fig.savefig('images/01_01_example_before.png')

    # Nakon transformacije
    img = images.transform(img, M=np.eye(2), T=np.array([[100, 200]]).T)
    fig = images.plot_image(img)
    fig.savefig('images/01_02_example_after.png')


if __name__ == '__main__':
    main()
