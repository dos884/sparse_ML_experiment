import tensorflow as tf
from sparsezoo import Zoo


def trash_generator(batch_size: int = 16, num_steps: int = None):
    i = 0
    if num_steps is None:
        num_steps = -1
    while i != num_steps:
        xs = tf.random.uniform([batch_size, 224, 224, 3])
        ys = tf.nn.softmax(tf.random.uniform([batch_size, 1000]), axis=-1)
        yield xs, ys
        i += 1


x = tf.constant(3)
x+x
print(tf.__version__)
yum_yum_trash_model = tf.keras.applications.ResNet50()
from sparseml.keras.optim import ScheduledModifierManager

manager = ScheduledModifierManager.from_yaml("recipe.yaml")

optimizer = tf.keras.optimizers.SGD()

sm_model, sm_optimizer, sm_callbacks = manager.modify(yum_yum_trash_model, optimizer, 500)
sm_model.compile(optimizer=sm_optimizer,
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=tf.keras.metrics.CategoricalAccuracy())
sm_model.fit(x=trash_generator(), callbacks= sm_callbacks,
          steps_per_epoch=500, epochs=3,
          validation_data=trash_generator(),
          validation_steps=500)


manager.finalize(sm_model)
