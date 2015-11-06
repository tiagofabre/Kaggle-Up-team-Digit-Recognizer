
sobrep = zeros(28,28,10);

for j=1:size(imgs,3)
    indice = targetValuesKaggle(j)+1;
    sobrep(:,:,indice) = (plus( sobrep(:,:,indice), imgs(:,:,j)));
end

for j=1:10
    sobrep(:,:,j) = (((sobrep(:,:,j)/42000)*10)*255);
end