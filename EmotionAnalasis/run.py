import test
import function
import embedding

if __name__=='__main__':
    order = input('choose embedding/train/test\n')
    if order=='embedding':
        embedding.build_embed()
    elif order=='train':
        function.train()
    elif order=='test':
        function.test()