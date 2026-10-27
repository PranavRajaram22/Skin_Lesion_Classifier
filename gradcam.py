import tensorflow as tf
import matplotlib.pyplot as plt
from data import test, classes, path_to_image

#paths that i want to analyse
correct_mel = ['skin-cancer-mnist-ham10000/versions/2/HAM10000_images_part_2/ISIC_0033426.jpg' ,'skin-cancer-mnist-ham10000/versions/2/HAM10000_images_part_2/ISIC_0033023.jpg', 'skin-cancer-mnist-ham10000/versions/2/HAM10000_images_part_1/ISIC_0025996.jpg']
missed_mel = ['skin-cancer-mnist-ham10000/versions/2/HAM10000_images_part_1/ISIC_0024987.jpg', 'skin-cancer-mnist-ham10000/versions/2/HAM10000_images_part_2/ISIC_0034087.jpg', 'skin-cancer-mnist-ham10000/versions/2/HAM10000_images_part_2/ISIC_0033027.jpg']
false_mel = ['skin-cancer-mnist-ham10000/versions/2/HAM10000_images_part_1/ISIC_0027466.jpg', 'skin-cancer-mnist-ham10000/versions/2/HAM10000_images_part_2/ISIC_0033821.jpg', 'skin-cancer-mnist-ham10000/versions/2/HAM10000_images_part_2/ISIC_0033132.jpg']
random_nv = ['skin-cancer-mnist-ham10000/versions/2/HAM10000_images_part_2/ISIC_0032296.jpg', 'skin-cancer-mnist-ham10000/versions/2/HAM10000_images_part_2/ISIC_0030633.jpg', 'skin-cancer-mnist-ham10000/versions/2/HAM10000_images_part_2/ISIC_0030597.jpg']



model = tf.keras.models.load_model('best_ft2.keras')
base  = model.get_layer('efficientnetv2-b1')
gap   = model.get_layer('global_average_pooling2d')
drop  = model.get_layer('dropout')
dense = model.get_layer('dense')

def gradcam(img, class_idx=None):          # img dims are 240x240x3, values are 0-255
    x = tf.convert_to_tensor(img[None], tf.float32)
    with tf.GradientTape() as tape:
        A = base(x, training=False)        # (1, 8, 8, 1280)
        tape.watch(A)
        p = dense(drop(gap(A), training=False))
        if class_idx is None:
            class_idx = int(tf.argmax(p[0]))
        score = p[0, class_idx]
    grads = tape.gradient(score, A)
    weights = tf.reduce_mean(grads, axis=(1, 2))                    # weight per channel
    cam = tf.nn.relu(tf.reduce_sum(A * weights[:, None, None, :], axis=-1))[0]
    return (cam / (tf.reduce_max(cam) + 1e-8)).numpy(), class_idx

mel = classes.index('mel')

def show(path, group):
    raw_bytes = tf.io.read_file(path)
    img = tf.io.decode_jpeg(raw_bytes, channels=3)
    img = tf.image.resize(img, (240, 240)).numpy()
    cam, c = gradcam(img)
    big = tf.image.resize(cam[..., None], (240, 240)).numpy()[..., 0]
    name = path.split('/')[-1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    ax1.imshow(img.astype('uint8'))
    ax1.set_title('original')
    ax1.axis('off')

    ax2.imshow(img.astype('uint8'))
    ax2.imshow(big, cmap='jet', alpha=0.4)
    ax2.set_title(f"Grad-CAM: {classes[c]}")
    ax2.axis('off')

    fig.suptitle(f"{group} | {name}")
    fig.savefig(f'gradcam_images/{group}_{name}.png', dpi=150)
    plt.close(fig)

groups = {'correct_mel': correct_mel, 'missed_mel': missed_mel,
          'false_mel': false_mel, 'random_nv': random_nv}

for group, paths in groups.items():
    for path in paths:
        show(path, group)

