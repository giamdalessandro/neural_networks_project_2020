#################################################################


# Codice python tradotto da quello demo di matlab per pascalvoc #


#################################################################




def VOClabelcolormap():
    # VOCLABELCOLORMAP Creates a label color map such that adjacent indices have different
    # colors.  Useful for reading and writing index images which contain large indices,
    # by encoding them as RGB images.
    # CMAP = VOCLABELCOLORMAP(N) creates a label color map with N entries.

    N = 256
    cmap = zeros(N,3)
    for i in range(1,N):
        id = i-1
        r=0
        g=0
        b=0
        for j=0:7
            r = bitor(r, bitshift(bitget(id,1),7 - j))
            g = bitor(g, bitshift(bitget(id,2),7 - j))
            b = bitor(b, bitshift(bitget(id,3),7 - j))
            id = bitshift(id,-3)
        end
        cmap(i,1)=r cmap(i,2)=g cmap(i,3)=b
    end
    cmap = cmap / 255


anno_files = './dataset/pascalvocpart/Annotations_Part/%s.mat'
examples_path = './dataset/pascalvocpart/examples'
examples_imgs = dir([examples_path, '/', '*.jpg'])
cmap = VOClabelcolormap()

pimap = part2ind()     #part index mapping

for ii = 1:numel(examples_imgs)
    imname = examples_imgs(ii).name
    img = imread([examples_path, '/', imname])
    % load annotation -- anno
    load(sprintf(anno_files, imname(1:end-4)))
    
    [cls_mask, inst_mask, part_mask] = mat2map(anno, img, pimap)
    
    % display annotation
    subplot(2,2,1) imshow(img) title('Image')
    subplot(2,2,2) imshow(cls_mask, cmap) title('Class Mask')
    subplot(2,2,3) imshow(inst_mask, cmap) title('Instance Mask')
    subplot(2,2,4) imshow(part_mask, cmap) title('Part Mask')
    pause
end