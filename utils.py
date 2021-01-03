import matplotlib.pyplot as plt


def plt_save(image, title=''):
    plt.imshow(X=image, cmap='gray')
    plt.title(title)
    file_path = './output/' + title + '.png'
    plt.savefig(file_path)
    print(file_path)
    pass


def plt_show(image, title=''):
    plt.imshow(X=image, cmap='gray')
    plt.title(title)
    plt.show()
