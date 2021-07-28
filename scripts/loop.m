myDir = ".\HEA12D_dataset\Asphalt_pavement_VD\Raw_images";	% gets directory
saveDir=".\Model4\Asphalt_pavement_VD\";
myFiles = dir(fullfile(myDir,'*.jpg'));	% Added closing parenthese!

for k = 1:length(myFiles)
	baseFileName = myFiles(k).name;
	fullFileName = fullfile(myDir, baseFileName);  % Changed myFolder to myDir
    fullFileSaveName= fullfile(saveDir, baseFileName);
    crack = imread(fullFileName);
    I9 = .2989*crack(:,:,1)+.5870*crack(:,:,2)+.1140*crack(:,:,3);
    % Identifying RGB components, square sizes
    crack1 = crack(:,:,1);
    crack2 = crack(:,:,2);
    crack3 = crack(:,:,3);
    I = medfilt2(I9);
    [a,b] = size(I);
    A1 = zeros(1,64);
    A = zeros(1,64);
    B = zeros(1,64);
    red = zeros(1,64);
    green = zeros(1,64);
    blue = zeros(1,64);
    % Identifying the size of squares depending on the image
    if((a >300) && (b>300))
     sx = 8;
     sy = 8;
    elseif((a>256) && (b>256))
     sx = 5;
     sy = 5;
    else
     sx = 6;
     sy = 6;
    end
    % Initialize variables to calculate the total of R, G and B components
    thresh = zeros(a/sx,b/sy);
    thresRGB = zeros(a/sx,b/sy);
    pixels = a*b;
    l=1;
    m=1;
    x=0;
    y=0;
    thres1 = 0;
    thres = 0;
    thresR = 0;
    thresG = 0;
    thresB = 0;
    I1 = I;
    ss = sx * sy ;
    % Background separation
    for x =0:sx:a
     for y = 0:sy:b
     for i=1:sx
     for j=1:sy
     for k = 1:ss
     if(((x+i) <= a) && ((y+j) <= b))
     A(1,k) = I(x+i,y+j);
     red(1,k) = crack1(x+i,y+j);
     green(1,k) = crack2(x+i,y+j);
     blue(1,k) = crack3(x+i,y+j);
     end
     end
     end
     k = 1;
     end
    % Calculate the mean of R, G and B components in each node
     for i=1:ss
     thres = thres + A(1,i);
     thresR = thresR + red(1,i);
     thresG = thresG + green(1,i);
     thresB = thresB + blue(1,i);
     end
     threshold = thres/ss;
     thresholdR = thresR/ss;
     thresholdG = thresG/ss;
     thresholdB = thresB/ss;
     thres = 0;
     thresR = 0;
     thresG = 0;
     thresB = 0;
     thresh(l,m) = threshold;
    % Calculate the standard deviation between means of each component
     De(1) = thresholdR;
     De(2) = thresholdG;
     De(3) = thresholdB;
     thresRGB(l,m) = std(De);
     m = m+1;
     m1 = m;
     end
     l = l+1;
     l1 = l;
     m = 1;
    end
    l=1;
    m=1;
    tval = 0;
    [rnum,cnum] = size(thresh);
    for i = 1 : rnum
     for j = 1 : cnum
     tval = tval + thresh(i,j);
     end
    end
    size21 = rnum * cnum;
    finthres = tval / size21;
    % Whiten the nodes that have a high standard deviation
    for g = 1:rnum
     for h = 1:cnum
     if((thresRGB(g,h) > 10) && (thresh(g,h)> finthres))
     for i = 1:sx
     for j = 1:sy
     I((((g-1)*sx)+i),(((h-1)*sy)+j)) = 255;
     end
     end
     end
     end
    end
    %figure,imshow(I,[]);
    % Verify for continuity of background
    for x = 1:rnum
     for y = 1:cnum
     count = 0;

     for i = -2:2
     for j = -2:2
     if((x+i <=rnum) && (y+j <=cnum) && ((x+i) >= 1) && ((y+j) >= 1))
     if((thresRGB(x+i,y+j)) > 10 && (thresh(x,y) > finthres))
     count = count + 1;
     end
     end
     end
     end
     % If the number of neighboring nodes that are white is more than a certain value, then 
    erase the node 
     if(count>2)
     for i = 1:sx
     for j = 1:sy
     I((((x-1)*sx)+i),(((y-1)*sy)+j)) = 255;
     end
     end
     end
     end
    end
    %figure,imshow(I,[]);
    % Contrasting
    % Identifying the value by which the intensities of pixels are changed
    if (finthres >= 120)
     changeval = 50;

    elseif (finthres >= 110 && finthres < 120)
     changeval = 45;
    elseif (finthres > 100 && finthres < 110)
     changeval = 40;
    elseif (finthres < 100 && finthres >= 90)
     changeval = 30;
    elseif (finthres < 90 && finthres >=75)
     changeval = 20;
    elseif (finthres <75 && finthres >= 60);
     changeval = 15;
    elseif (finthres < 60)
     changeval = 10;
    end
    % Altering the values of pixels
    for x = 1:a-sx
     for y = 1:b-sy
     for i=0:sx
     for j=0:sy
     if I(x+i,y+j) <= thresh(l,m)
     I1(x+i,y+j) = I(x+i,y+j) - changeval;
     else
     I1(x+i,y+j) = I(x+i,y+j) + changeval;
     end
     m = m+1;
     end
     l = l+1;
     m = 1;
     end
     l = 1;
     end
    end
    %figure,imshow(I1,[]); %3
    %Verify maximum permissible threshold for intensity
    [number,intensity] = imhist(I1);
    RI = a * b / 10;

    numbers = 0;
    for i = 1:255
     numbers = numbers + number(i);
     if(numbers < RI)
     intbord = i;
     end
    end
    intbord = intbord/255;
    % Localized thresholds after contrasting are identified 
    for x =0:sx:a
     for y = 0:sy:b
     for i=1:sx
     for j=1:sy
     for k = 1:ss
     if(((x+i) <= a) && ((y+j) <= b))
     A1(1,k) = I1(x+i,y+j);
     end
     end
     end
     k = 1;
     end
     for i=1:ss
     thres1 = thres1 + A1(1,i);
     end
     threshold1 = thres1/36;
     thres1 = 0;
     thresh1(l,m) = threshold1;
     m = m+1;
     m1 = m;
     end
     l = l+1;
     l1 = l;
     m = 1;
    end
    l=1;
    m=1;
    tval1 = 0;
    for i = 1 : rnum
     for j = 1 : cnum
     if(thresh1(i,j) ~= 255)
     tval1 = tval1 + thresh1(i,j);
     end
     end
    end
    size21 = rnum * cnum;
    finthres1 = tval1 / size21;

    if (finthres1 > finthres)
     finthres1 = finthres;
    end
    % Mean value of the intensities in region of interest is used to find a threshold to convert 
    %image to black and white 
    if (finthres1 >= 120)
     bord = 0.17;
    elseif (finthres1 >= 110 && finthres1 < 120)
     bord = 0.13;
    elseif (finthres1 > 100 && finthres1 < 110)
     bord = 0.11;
    elseif (finthres1 < 100 && finthres1 >= 90)
     bord = finthres1 / 4;
    elseif (finthres1 < 90 && finthres1 >= 75)
     bord = 0.15;
    elseif (finthres1 < 75)
     bord = 0.05;
    end
    while (bord > intbord)
     bord = bord * 0.75;
    end
    %Convert to black and white image
    I2 = im2bw(I1,bord);
    %figure,imshow(I2,[]); %4
    for i= 1:a
     for j=1:b
     I3(i,j) = 1 - I2(i,j);
     end
    end
    %figure,imshow(I3,[]); %5
    % Verify for continuity by verifying the neighborhood of pixels
    for x = 3:a
     for y = 3:b
     count = 0;
     if(I3(x,y) == 0)
     for i = -2:2
     for j = -2:2
     if( ( (x+i) <= a ) && ((x+i) >= 1) && ( (y+j) <= b ) && ((y+j) >=1) ) 
     if(I3(x+i,y+j) == 1)
     count = count + 1;
     end
     end
     end
     end
     end
     if(count>2)

    I4(x,y) = 1;
     else
     I4(x,y) = I3(x,y);
     end
     end
    end
    imwrite(I4,fullFileSaveName);
    end
