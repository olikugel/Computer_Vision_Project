#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <jpeglib.h>
#include <math.h>
#define PI 3.1415926535898

// -------------------------------------------------------------------------------

#define CALL_LabelComponent(x,y,returnLabel) { STACK[SP] = x; STACK[SP+1] = y; STACK[SP+2] = returnLabel; SP += 3; goto START; }

#define RETURN { SP -= 3;                \
                 switch (STACK[SP+2])    \
                 {                       \
                 case 1 : goto RETURN1;  \
                 case 2 : goto RETURN2;  \
                 case 3 : goto RETURN3;  \
                 case 4 : goto RETURN4;  \
                 default: return;        \
                 }                       \
               }

#define X (STACK[SP-3])
#define Y (STACK[SP-2])

int maxBrightness = 256;

// -------------------------------------------------------------------------------






/** Read the JPEG image at `filename` as an array of bytes.
  Data is returned through the out pointers, while the return
  value indicates success or failure.
  NOTE: 1) if image is RGB, then the bytes are concatenated in R-G-B order
        2) `image` should be freed by the user
 */
static inline int
read_JPEG_file(char *filename,
               int *width, int *height, int *channels, unsigned char *(image[]))
{
  FILE *infile;
  if ((infile = fopen(filename, "rb")) == NULL) {
    fprintf(stderr, "can't open %s\n", filename);
    return 0;
  }

  struct jpeg_error_mgr jerr;
  struct jpeg_decompress_struct cinfo;
  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_decompress(&cinfo);
  jpeg_stdio_src(&cinfo, infile);
  (void) jpeg_read_header(&cinfo, TRUE);
  (void) jpeg_start_decompress(&cinfo);

  *width = cinfo.output_width, *height = cinfo.output_height;
  *channels = cinfo.num_components;
  // printf("width=%d height=%d c=%d\n", *width, *height, *channels);
  *image = malloc(*width * *height * *channels * sizeof(*image));
  JSAMPROW rowptr[1];
  int row_stride = *width * *channels;

  while (cinfo.output_scanline < cinfo.output_height) {
    rowptr[0] = *image + row_stride * cinfo.output_scanline;
    jpeg_read_scanlines(&cinfo, rowptr, 1);
  }
  jpeg_finish_decompress(&cinfo);

  jpeg_destroy_decompress(&cinfo);
  fclose(infile);
  return 1;
}






/** Writes the image in the specified file.
  NOTE: works with Grayscale or RGB modes only (based on number of channels)
 */
static inline void
write_JPEG_file(char *filename, int width, int height, int channels,
                unsigned char image[], int quality)
{
  FILE *outfile;
  if ((outfile = fopen(filename, "wb")) == NULL) {
    fprintf(stderr, "can't open %s\n", filename);
    exit(1);
  }

  struct jpeg_error_mgr jerr;
  struct jpeg_compress_struct cinfo;
  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_compress(&cinfo);
  jpeg_stdio_dest(&cinfo,outfile);

  cinfo.image_width = width;
  cinfo.image_height = height;
  cinfo.input_components = channels;
  cinfo.in_color_space = channels == 1 ? JCS_GRAYSCALE : JCS_RGB;
  jpeg_set_defaults(&cinfo);
  jpeg_set_quality(&cinfo, quality, TRUE);

  jpeg_start_compress(&cinfo, TRUE);
  JSAMPROW rowptr[1];
  int row_stride = width * channels;
  while (cinfo.next_scanline < cinfo.image_height) {
    rowptr[0] = & image[cinfo.next_scanline * row_stride];
    jpeg_write_scanlines(&cinfo, rowptr, 1);
  }
  jpeg_finish_compress(&cinfo);

  fclose(outfile);
  jpeg_destroy_compress(&cinfo);
}





void findLabel(unsigned short* STACK, int width, int height, unsigned char* sourceImage, int* labelledImage, int labelNumber, int j, int i)
{
  STACK[0] = j;
  STACK[1] = i;
  STACK[2] = 0;  // return, component is labelled
  int SP   = 3;
  int index;

  START: // recursive routine starts here

      index = X + width * Y;
      if (sourceImage [index] == 0) RETURN;   // pixel is not part of a component
      if (labelledImage[index] != 0) RETURN;  // pixel has already been labelled
      labelledImage[index] = labelNumber;

      if (X > 0) CALL_LabelComponent(X-1, Y, 1);        // WEST
        RETURN1:

      if (X < width-1) CALL_LabelComponent(X+1, Y, 2);  // EAST
        RETURN2:

      if (Y > 0) CALL_LabelComponent(X, Y-1, 3);        // NORTH
        RETURN3:

      if (Y < height-1) CALL_LabelComponent(X, Y+1, 4); // SOUTH
        RETURN4:

  RETURN;

} // findLabel()






void assignLabels(int width, int height, unsigned char* sourceImage, int* labelledImage)
{
  int size = width * height;
  unsigned short* STACK = (unsigned short*) malloc(3 * sizeof(unsigned short) * (size + 1));

  int labelNumber = 40; // set the brightness of the first component here
  int index = -1;

  for (int i = 0; i < height; i++)
    for (int j = 0; j < width; j++)
    {
      index++;
      if (sourceImage [index] == 0) continue;   // pixel is not part of a component
      if (labelledImage[index] != 0) continue;  // pixel has already been labelled
      // New component was found
      labelNumber += 2;
      findLabel(STACK, width, height, sourceImage, labelledImage, labelNumber, j, i);
    }

  free(STACK);

} // assignLabels()






int findThreshold(int* histogram)
{
  int currentThreshold;
  int currentPixel;

  int entriesLeft;
  int entriesRight;

  int accumLeft;
  int accumRight;

  int finalThreshold = -1; // -1 used for debugging

  double leftAverage  = 0;
  double rightAverage = 0;
  double averageOfLeftAndRight = 0;


  for (currentThreshold = 0; currentThreshold < maxBrightness; currentThreshold++)
  {
    accumLeft    = 0;
    accumRight   = 0;
    entriesLeft  = 0;
    entriesRight = 0;

    // accumulate the pixel values of all pixels to the left of (and including) the threshold
    for (currentPixel = 0; currentPixel <= currentThreshold; currentPixel++)
    {
      entriesLeft += histogram[currentPixel];
      accumLeft += (histogram[currentPixel] * currentPixel);
    }

    // accumulate the pixel values of all pixels to the right of the threshold
    for (currentPixel = currentThreshold + 1; currentPixel < maxBrightness; currentPixel++)
    {
      entriesRight += histogram[currentPixel];
      accumRight += (histogram[currentPixel] * currentPixel);
    }

    if (entriesLeft != 0  &&  entriesRight != 0) // we don't want to divide by zero
    {
      leftAverage  = (double)accumLeft  / (double)entriesLeft;
      rightAverage = (double)accumRight / (double)entriesRight;

      averageOfLeftAndRight = (leftAverage + rightAverage) / 2;
      // printf("Current Threshold: %d\tL&R Average: %f\n", currentThreshold, averageOfLeftAndRight);

      if (round(averageOfLeftAndRight) == currentThreshold)
      {
        finalThreshold = currentThreshold;
        return finalThreshold;
      } // if
    } // if

  } // outer for

  return finalThreshold;

} // findThreshold()






unsigned char* computeThresholdedImage(unsigned char* image, int width, int height, int threshold)
{
  int size = width * height;
  unsigned char* thresholdedImage = (unsigned char*) malloc(size * sizeof(unsigned char));

  for(int i = 0; i < height; i++)
    for (int j = 0; j < width; j++)
    {
      if (image[i * width + j] <= threshold)
        thresholdedImage[i * width + j] = 255; // below or on threshold --> white
      else
        thresholdedImage[i * width + j] = 0; // above threshold --> black
    } // for

  return thresholdedImage;

} // computeThresholdedImage()






int compareFunction(const void * a, const void * b)
{
   return ( *(int*)a - *(int*)b );
}





int findMedian5(unsigned char* image, int pixelIndex, int imageWidth)
{
  int i = pixelIndex;
  int w = imageWidth;
  int templateSize = 5 * 5;
  int* fiveByFive = (int*) malloc(templateSize * sizeof(int));

  fiveByFive[0]  = image[i-2-(2*w)];
  fiveByFive[1]  = image[i-1-(2*w)];
  fiveByFive[2]  = image[i-(2*w)];
  fiveByFive[3]  = image[i+1-(2*w)];
  fiveByFive[4]  = image[i+2-(2*w)];
  fiveByFive[5]  = image[i-2-w];
  fiveByFive[6]  = image[i-1-w];
  fiveByFive[7]  = image[i-w];
  fiveByFive[8]  = image[i+1-w];
  fiveByFive[9]  = image[i+2-w];
  fiveByFive[10] = image[i-2];
  fiveByFive[11] = image[i-1];
  fiveByFive[12] = image[i];
  fiveByFive[13] = image[i+1];
  fiveByFive[14] = image[i+2];
  fiveByFive[15] = image[i-2+w];
  fiveByFive[16] = image[i-1+w];
  fiveByFive[17] = image[i+w];
  fiveByFive[18] = image[i+1+w];
  fiveByFive[19] = image[i+2+w];
  fiveByFive[20] = image[i-2+(2*w)];
  fiveByFive[21] = image[i-1+(2*w)];
  fiveByFive[22] = image[i+(2*w)];
  fiveByFive[23] = image[i+1+(2*w)];
  fiveByFive[24] = image[i+2+(2*w)];

  qsort(fiveByFive, templateSize, sizeof(int), compareFunction);

  int median = fiveByFive[12];
  free(fiveByFive);
  return median;

} //  findMedian5()






unsigned char* computeSmoothedImage(unsigned char* image, int width, int height, int edge)
{
  int size = width * height;
  int pixelIndex;
  unsigned char* smoothedImage = (unsigned char*) malloc(size * sizeof(unsigned char));

  // copy everything over from the source image, boundaries will be preserved
  for (int i = 0; i < size; i++)
    smoothedImage[i] = image[i];

  for (int i = edge; i < height - edge; i++)
    for(int j = edge; j < width - edge; j++)
    {
      pixelIndex = i * width + j;
      smoothedImage[pixelIndex] = (unsigned char) findMedian5(image, pixelIndex, width);
    }

  return smoothedImage;

} // computeSmoothedImage()





int* constructHorizontalConvolMatrix(int matrixSize)
{
  // this is the 1-D array that will store the 2-D horizontal convolution matrix
  // values are stored from left to right, row by row
  int* horizontalConvolMatrix = (int*) malloc(matrixSize * sizeof(int));

  horizontalConvolMatrix[0] = -1;
  horizontalConvolMatrix[1] =  0;
  horizontalConvolMatrix[2] =  1;
  horizontalConvolMatrix[3] = -2;
  horizontalConvolMatrix[4] =  0;
  horizontalConvolMatrix[5] =  2;
  horizontalConvolMatrix[6] = -1;
  horizontalConvolMatrix[7] =  0;
  horizontalConvolMatrix[8] =  1;

  return horizontalConvolMatrix;

} // constructHorizontalConvolMatrix()





int* constructVerticalConvolMatrix(int matrixSize)
{
  // this is the 1-D array that will store the 2-D vertical convolution matrix
  // values are stored from left to right, row by row
  int* verticalConvolMatrix = (int*) malloc(matrixSize * sizeof(int));

  verticalConvolMatrix[0] = -1;
  verticalConvolMatrix[1] = -2;
  verticalConvolMatrix[2] = -1;
  verticalConvolMatrix[3] =  0;
  verticalConvolMatrix[4] =  0;
  verticalConvolMatrix[5] =  0;
  verticalConvolMatrix[6] =  1;
  verticalConvolMatrix[7] =  2;
  verticalConvolMatrix[8] =  1;

  return verticalConvolMatrix;

} // constructHorizontalConvolMatrix()





int convolve(unsigned char* sourceImage, int pixelIndex, int imageWidth, int* convolMatrix)
{
  int i = pixelIndex;
  int w = imageWidth;
  int result;

  result =   sourceImage[(i-1)-w] * convolMatrix[0]
           + sourceImage[i-w]     * convolMatrix[1]
           + sourceImage[(i+1)-w] * convolMatrix[2]
           + sourceImage[i-1]     * convolMatrix[3]
           + sourceImage[i]       * convolMatrix[4]
           + sourceImage[i+1]     * convolMatrix[5]
           + sourceImage[(i-1)+w] * convolMatrix[6]
           + sourceImage[i+w]     * convolMatrix[7]
           + sourceImage[(i+1)+w] * convolMatrix[8];

  return result;
} // convolve()





int* computeEdgeResponseImage(unsigned char* sourceImage, int* convolMatrix,
                              int width, int height, int edge)
{
  int size = width * height;
  int* edgeResponseImage = (int*) malloc(size * sizeof(int));
  int pixelIndex;

  // Copy everything from sourceImage to responseImage, boundaries will be preserved
  for (int i = 0; i < size; i++)
    edgeResponseImage[i] = sourceImage[i];

  // Convolve the sourceImage with the convolMatrix, for every pixel except the ones on boundaries
  for (int y = edge; y < height - edge; y++)
    for(int x = edge; x < width - edge; x++)
    {
      pixelIndex = y * width + x;
      edgeResponseImage[pixelIndex] = convolve(sourceImage, pixelIndex, width, convolMatrix);
    }

  return edgeResponseImage;
} // computeEdgeResponseImage()





unsigned char* computeEdgedImage(int* horizontalEdgeResponseImage,
                                 int* verticalEdgeResponseImage, int size)
{
  unsigned char* edgedImage = (unsigned char*) malloc(size * sizeof(unsigned char));

  for (int i = 0; i < size; i++)
    edgedImage[i] = sqrt(  pow(horizontalEdgeResponseImage[i], 2)
                         + pow(verticalEdgeResponseImage[i], 2));

  return edgedImage;

} // computeEdgedImage()





unsigned char* computeOrientationImage(int* horizontalEdgeResponseImage,
                                       int* verticalEdgeResponseImage,
                                       int size)
{
  double intermedResult;
  unsigned char* orientationImage = (unsigned char*) malloc(size * sizeof(unsigned char));

  for (int i = 0; i < size; i++)
  {
    intermedResult = atan2(verticalEdgeResponseImage[i], horizontalEdgeResponseImage[i]);
    orientationImage[i] = (unsigned char) (((intermedResult + PI) / (2 * PI)) * 255);
  }

  return orientationImage;

} // computeGradientMagnitudeImage()





unsigned char* computeLabelledImage(unsigned char* image, int width, int height)
{
  int size = width * height;

  int* labelledImage = (int*) malloc(size * sizeof(int));

  memset(labelledImage, 0, size * sizeof(int));

  assignLabels(width, height, image, labelledImage);

  unsigned char* labelledImageAsChars = (unsigned char*) malloc(size * sizeof(unsigned char));

  for (int i = 0; i < size; i++)
    labelledImageAsChars[i] = (unsigned char) labelledImage[i];

  free(labelledImage);

  return labelledImageAsChars;

} // computeLabelledImage()






int* createHistogram(unsigned char* image, int width, int height)
{
  int currentBrightness;
  int* histogram = (int*) malloc(maxBrightness * sizeof(int));

  // initialise histogram to zero
  for(int i = 0; i < maxBrightness; i++)
    histogram[i] = 0;

  // loop through image, increment brightnesses
  for(int i = 0; i < height; i++)
    for (int j = 0; j < width; j++)
    {
      currentBrightness = image[i * width + j];
      histogram[currentBrightness]++;
    }

  return histogram;

} // createHistogram()





void printHistogram(int* histogram)
{
  // print histogram
  for(int i = 0; i < maxBrightness; i++)
    if (histogram[i] != 0)
      printf("histogram[%d]\t= %d\n", i, histogram[i]);
} // printHistogram()





int* createCumulativeHistogram(int* histogram, int width, int height)
{
  int* cumulativeHistogram = (int*) malloc(maxBrightness * sizeof(int));

  for(int i = 0; i < maxBrightness; i++)
    cumulativeHistogram[i] = 0;

  for(int i = 0; i < maxBrightness; i++)
    for(int j = 0; j < maxBrightness; j++)
      if (j <= i)
        cumulativeHistogram[i] += histogram[j];

  // print cumulative histogram
  // printf("\n\nCumulative Histogram\n");
  // for(int i = 0; i < maxBrightness; i++)
  //   printf("c[%d] = %d\n", i, cumulativeHistogram[i]);

  return cumulativeHistogram;

} // createHistogram()





int* createLookupTable(int* cumulativeHistogram, int width, int height)
{
  double numberOfPixels = width * height;
  double numberOfGreyValues = maxBrightness;
  double* unroundedLookupTable = (double*) malloc(maxBrightness * sizeof(double));
  int* lookupTable = (int*) malloc(maxBrightness * sizeof(int));

  // initialise lookup table all to zero
  for(int i = 0; i < maxBrightness; i++)
  {
    unroundedLookupTable[i] = 0;
    lookupTable[i] = 0;
  }

  for(int j = 0; j < maxBrightness; j++)
    unroundedLookupTable[j] = (numberOfGreyValues / numberOfPixels) * (double) cumulativeHistogram[j];

  for(int i = 0; i < maxBrightness; i++)
  {
    lookupTable[i] = (int) round(unroundedLookupTable[i]);

    if (lookupTable[i] > 255)
      lookupTable[i] = 255;
    if (lookupTable[i] < 0)
      lookupTable[i] = 0;
  }

  /*
  printf("\nLookup Table\n\n");
  for(int i = 0; i < maxBrightness; i++)
    if (lookupTable[i] != 0)
      printf("lookup[%d] = %d\n", i, lookupTable[i]);
  */

  free(unroundedLookupTable);
  return lookupTable;

} // createLookupTable()





unsigned char* equaliseImage(unsigned char* image, int* lookupTable, int width, int height)
{
  int size = width * height;
  unsigned char* equalisedImage = (unsigned char*) malloc(size * sizeof(unsigned char));

  for(int i = 0; i < height; i++)
    for (int j = 0; j < width; j++)
      equalisedImage[i * width + j] = lookupTable[image[i * width + j]];

  return equalisedImage;
} // equaliseImage()





unsigned char* computeClassifiedImage(unsigned char* image, int* histogram, int width, int height)
{
  int classificationThreshold = 1225;
  int size = width * height;
  int pixelIndex;
  unsigned char* classifiedImage = (unsigned char*) malloc(size * sizeof(unsigned char));

  for(int i = 0; i < height; i++)
    for (int j = 0; j < width; j++)
    {
      pixelIndex =  i * width + j;

      if (image[pixelIndex] != 0)  // ignore background
      {
        if (histogram[image[pixelIndex]] > classificationThreshold)
          classifiedImage[pixelIndex] = 255; // NORMAL CELL
        else
          classifiedImage[pixelIndex] = 110; // ABNORMAL CELL
      }
    } // inner-for

  return classifiedImage;
}






int main(int argc, char *argv[])
{
  if (argc != 10)
  {
    printf("\nUsage:\n./ImageProcessing sickle1.jpg thresholded.jpg equalised.jpg smoothed1.jpg smoothed2.jpg edged.jpg orientation.jpg labelled.jpg classified.jpg\n");
    printf("\nOr better:\nmake compile\nmake run\n\n");
    return 1;
  }

//--------------------------------------------------------------------------------------

  unsigned char* sourceImage;
  unsigned char* thresholdedImage;
  unsigned char* equalisedImage;
  unsigned char* smoothed1Image;
  unsigned char* smoothed2Image;
  unsigned char* edgedImage;
  unsigned char* orientationImage;
  unsigned char* labelledImage;
  unsigned char* classifiedImage;

  int* horizontalEdgeResponseImage;
  int* verticalEdgeResponseImage;

  int* horizontalConvolMatrix;
  int* verticalConvolMatrix;

  int* histogram;
  int* histogramOfLabelledImage;
  int* histogramAfterEqualising;
  int* cumulativeHistogram;
  int* lookupTable;

  int width;
  int height;
  int channels; // ignore this because we're dealing with grey-scale images
  int threshold;

  int convolMatrixWidth  = 3;
  int convolMatrixHeight = 3;
  int convolMatrixSize = convolMatrixWidth * convolMatrixHeight;
  int edge = convolMatrixWidth / 2; // deliberate integer division

  int medianFilterTemplateWidth = 5;
  int medianEdge = medianFilterTemplateWidth / 2; // deliberate integer division

  read_JPEG_file(argv[1], &width, &height, &channels, &sourceImage);

  int size = width * height;

//--------------------------------------------------------------------------------------

  histogram = createHistogram(sourceImage, width, height);

  // printHistogram(histogram);

  threshold = findThreshold(histogram);

  printf("\nThreshold: %d\n\n", threshold);

  thresholdedImage = computeThresholdedImage(sourceImage, width, height, threshold);

//--------------------------------------------------------------------------------------

  cumulativeHistogram = createCumulativeHistogram(histogram, width, height);

  // printHistogram(cumulativeHistogram);

  lookupTable = createLookupTable(cumulativeHistogram, width, height);

  equalisedImage = equaliseImage(sourceImage, lookupTable, width, height);

  histogramAfterEqualising = createHistogram(equalisedImage, width, height);

  // printHistogram(histogramAfterEqualising);

//--------------------------------------------------------------------------------------

  horizontalConvolMatrix = constructHorizontalConvolMatrix(convolMatrixSize);

  verticalConvolMatrix   = constructVerticalConvolMatrix(convolMatrixSize);

//--------------------------------------------------------------------------------------

  smoothed1Image = computeSmoothedImage(sourceImage, width, height, medianEdge);

  for (int i = 0; i < 12; i++)
    smoothed1Image = computeSmoothedImage(smoothed1Image, width, height, medianEdge);

//--------------------------------------------------------------------------------------

  horizontalEdgeResponseImage = computeEdgeResponseImage(smoothed1Image, horizontalConvolMatrix,
                                                         width, height, edge);

  verticalEdgeResponseImage = computeEdgeResponseImage(smoothed1Image, verticalConvolMatrix,
                                                       width, height, edge);

  edgedImage = computeEdgedImage(horizontalEdgeResponseImage, verticalEdgeResponseImage, size);

//--------------------------------------------------------------------------------------

  orientationImage = computeOrientationImage(horizontalEdgeResponseImage,
                                             verticalEdgeResponseImage, size);

//--------------------------------------------------------------------------------------

  smoothed2Image = computeSmoothedImage(thresholdedImage, width, height, medianEdge);

  for (int i = 0; i < 25; i++)
    smoothed2Image = computeSmoothedImage(smoothed2Image, width, height, medianEdge);

//--------------------------------------------------------------------------------------

  labelledImage = computeLabelledImage(smoothed2Image, width, height);

  histogramOfLabelledImage = createHistogram(labelledImage, width, height);

  printHistogram(histogramOfLabelledImage);

//--------------------------------------------------------------------------------------

  classifiedImage = computeClassifiedImage(labelledImage, histogramOfLabelledImage,
                                           width, height);

//--------------------------------------------------------------------------------------

  write_JPEG_file(argv[2], width, height, channels, thresholdedImage, 95);

  write_JPEG_file(argv[3], width, height, channels, equalisedImage, 95);

  write_JPEG_file(argv[4], width, height, channels, smoothed1Image, 95);

  write_JPEG_file(argv[5], width, height, channels, smoothed2Image, 95);

  write_JPEG_file(argv[6], width, height, channels, edgedImage, 95);

  write_JPEG_file(argv[7], width, height, channels, orientationImage, 95);

  write_JPEG_file(argv[8], width, height, channels, labelledImage, 95);

  write_JPEG_file(argv[9], width, height, channels, classifiedImage, 95);

//--------------------------------------------------------------------------------------

  printf("\n");
  free(thresholdedImage);
  free(equalisedImage);
  free(smoothed1Image);
  free(smoothed2Image);
  free(horizontalConvolMatrix);
  free(verticalConvolMatrix);
  free(horizontalEdgeResponseImage);
  free(verticalEdgeResponseImage);
  free(edgedImage);
  free(orientationImage);
  free(labelledImage);
  free(classifiedImage);
  free(histogram);

//--------------------------------------------------------------------------------------

  return 0;

} // main()
