from models import gan_cls, wgan_cls


class gan_factory(object):

    @staticmethod
    def generator_factory(type):
        if type == 'gan':
            return gan_cls.generator()
        elif type == 'wgan':
            return wgan_cls.generator()

    @staticmethod
    def discriminator_factory(type):
        if type == 'gan':
            return gan_cls.discriminator()
        elif type == 'wgan':
            return wgan_cls.discriminator()
