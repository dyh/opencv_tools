import matplotlib.pyplot as plt


def plt_show(image, title=''):
    plt.imshow(X=image, cmap='gray')
    plt.title(title)
    plt.show()

# def plt_show_all(images):
#     length1 = len(images)
#     plt.figure()
#     for i in range(0, length1):
#         plt.subplot(length1, 2, i+1)
#         plt.imshow(images[i])
#         plt.xticks([])
#         plt.yticks([])
#     plt.show()
