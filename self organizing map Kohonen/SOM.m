% self organizing map Kohonen
% Programmed by Ammar AL-Jodah
clc
clear all
close all
for i=1:1000
    x1(i)=rand;
    x2(i)=rand;
end
for j1=1:10
    for j2=1:10
        w1(j1,j2)=rand*(0.52-0.48)+0.48;
        w2(j1,j2)=rand*(0.52-0.48)+0.48;
    end
end
figure(1)
plot(x1,x2,'.b')
hold on
plot(w1,w2,'or')
plot(w1,w2,'k','linewidth',2)
plot(w1',w2','k','linewidth',2)
hold off
title('t=0');
drawnow
no=1;
do=5;
T=300;
t=1;
while (t<=T)
    n=no*(1-t/T);
    d=round(do*(1-t/T));
    %loop for the 1000 inputs
    for i=1:1000
        e_norm=(x1(i)-w1).^2+(x2(i)-w2).^2;
        minj1=1;minj2=1;
        min_norm=e_norm(minj1,minj2);
        for j1=1:10
            for j2=1:10
                if e_norm(j1,j2)<min_norm
                    min_norm=e_norm(j1,j2);
                    minj1=j1;
                    minj2=j2;
                end
            end
        end
        j1star= minj1;
        j2star= minj2;
        %update the winning neuron
        w1(j1star,j2star)=w1(j1star,j2star)+n*(x1(i)- w1(j1star,j2star));
        w2(j1star,j2star)=w2(j1star,j2star)+n*(x2(i)- w2(j1star,j2star));
        %update the neighbour neurons
        for dd=1:1:d
            jj1=j1star-dd;
            jj2=j2star;
            if (jj1>=1)
                w1(jj1,jj2)=w1(jj1,jj2)+n*(x1(i)-w1(jj1,jj2));
                w2(jj1,jj2)=w2(jj1,jj2)+n*(x2(i)-w2(jj1,jj2));
            end
            jj1=j1star+dd;
            jj2=j2star;
            if (jj1<=10)
                w1(jj1,jj2)=w1(jj1,jj2)+n*(x1(i)-w1(jj1,jj2));
                w2(jj1,jj2)=w2(jj1,jj2)+n*(x2(i)-w2(jj1,jj2));
            end
            jj1=j1star;
            jj2=j2star-dd;
            if (jj2>=1)
                w1(jj1,jj2)=w1(jj1,jj2)+n*(x1(i)-w1(jj1,jj2));
                w2(jj1,jj2)=w2(jj1,jj2)+n*(x2(i)-w2(jj1,jj2));
            end
            jj1=j1star;
            jj2=j2star+dd;
            if (jj2<=10)
                w1(jj1,jj2)=w1(jj1,jj2)+n*(x1(i)-w1(jj1,jj2));
                w2(jj1,jj2)=w2(jj1,jj2)+n*(x2(i)-w2(jj1,jj2));
            end
        end
    end
    t=t+1;
    figure(1)
    plot(x1,x2,'.b')
    hold on
    plot(w1,w2,'or')
    plot(w1,w2,'k','linewidth',2)
    plot(w1',w2','k','linewidth',2)
    hold off
    title(['t=' num2str(t)]);
    drawnow
    
end

