from utils import images
import numpy as np


def main():
    # Pre transformacije
    img_orig = images.load_image('doge.png')
    img_orig = images.add_axis(img_orig)
    fig = images.plot_image(img_orig)
    fig.savefig('images/02_01_example_before.png')

    # Nakon transformacije
    theta = np.sqrt(2) / 2
    img = images.transform(img_orig, M=np.array([[theta, -theta], [theta, theta]]))
    fig = images.plot_image(img)
    fig.savefig('images/02_02_example_after.png')

    # Nakon transformacije
    img_center = np.array([[img_orig.shape[0] // 2, img_orig.shape[1] // 2]]).T
    img = images.transform(img_orig, M=np.array([[theta, -theta], [theta, theta]]), trans_point=img_center)
    fig = images.plot_image(img)
    fig.savefig('images/02_03_example_after.png')


if __name__ == '__main__':
    main()
