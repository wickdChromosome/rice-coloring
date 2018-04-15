#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkBinaryImageToShapeLabelMapFilter.h"
#include "itkLabelMapToLabelImageFilter.h"
#include "itkBinaryImageToStatisticsLabelMapFilter.h"
#include "itkScalarImageToTextureFeaturesFilter.h"
#include "itkBinaryShapeOpeningImageFilter.h"
#include "itkCastImageFilter.h"
#include "itkBinaryImageToLabelMapFilter.h"
#include "itkBinaryImageToShapeLabelMapFilter.h"
#include "itkLabelMapOverlayImageFilter.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkBinaryThresholdImageFilter.h"
#include "itkBinaryMedianImageFilter.h"
#include "itkRGBPixel.h"
#include "itkMath.h"
#include "itkBinaryShapeOpeningImageFilter.h"
#include <vector>
#include "itkImageFileWriter.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkBinaryBallStructuringElement.h"
#include "itkBinaryErodeImageFilter.h"


#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkSigmoidImageFilter.h"
#include "itkAdaptiveHistogramEqualizationImageFilter.h"
#include "itkLaplacianSharpeningImageFilter.h"
#include "itkCastImageFilter.h"
#include "itkRescaleIntensityImageFilter.h"


using namespace std;


int main(int argc, char *argv[])
{

	char *inputFileName = "rice.png";


	int count, i;

	typedef short int ShortIntPixelType;
	typedef unsigned char UCHARPixelType;
	typedef itk::RGBPixel<unsigned char> RGBPixelType;

	typedef itk::Image<ShortIntPixelType, 2> ShortImageType;
	typedef itk::Image<UCHARPixelType, 2> UCharImageType;
	typedef itk::Image<RGBPixelType, 2> RGBImageType;

	typedef itk::ImageFileReader<UCharImageType> ReaderType;
	ReaderType::Pointer featureReader = ReaderType::New();
	featureReader->SetFileName(inputFileName);
	featureReader->Update();

	//make binary image
	typedef itk::BinaryThresholdImageFilter<UCharImageType, UCharImageType> ThresholdType;
	ThresholdType::Pointer thresholder = ThresholdType::New();
	thresholder->SetInput(featureReader->GetOutput());
	thresholder->SetInsideValue(255);
	thresholder->SetOutsideValue(0);
	thresholder->SetLowerThreshold(110);
	thresholder->SetUpperThreshold(255);
	thresholder->Update();

	//remove noise
	typedef itk::BinaryMedianImageFilter<UCharImageType, UCharImageType> FilterType;
	FilterType::Pointer filter = FilterType::New();
	ShortImageType::SizeType indexRadius;
	indexRadius[0] = 1;
	indexRadius[1] = 1;
	filter->SetRadius(indexRadius);
	filter->SetInput(thresholder->GetOutput());
	filter->Update();

	//erode grains
	typedef itk::BinaryBallStructuringElement<
	    UCharImageType::PixelType, 2>                  StructuringElementType;
	StructuringElementType structuringElement;
	structuringElement.SetRadius(2);
	structuringElement.CreateStructuringElement();
	typedef itk::BinaryErodeImageFilter <UCharImageType,UCharImageType, StructuringElementType>
    	BinaryErodeImageFilterType;

  	BinaryErodeImageFilterType::Pointer erodeFilter = BinaryErodeImageFilterType::New();
  	erodeFilter->SetInput(filter->GetOutput());
  	erodeFilter->SetKernel(structuringElement);

	//remove border grains
	typedef itk::BinaryShapeOpeningImageFilter<UCharImageType> OpeningType;
	OpeningType::Pointer openingFilter = OpeningType::New();
	openingFilter->SetInput(erodeFilter->GetOutput()); 
	openingFilter->SetForegroundValue(255);
	openingFilter->SetBackgroundValue(0);
	openingFilter->SetLambda(0);
	openingFilter->SetReverseOrdering(true);
	openingFilter->SetFullyConnected(true);
	openingFilter->SetAttribute("NumberOfPixelsOnBorder"); 
	openingFilter->Update();


	//Get size of objects
	typedef itk::BinaryShapeOpeningImageFilter<UCharImageType> OpeningType3;
	OpeningType3::Pointer openingFilter3 = OpeningType3::New();
	openingFilter3->SetInput(openingFilter->GetOutput()); 
	openingFilter3->SetForegroundValue(255);
	openingFilter3->SetBackgroundValue(0);
	openingFilter3->SetReverseOrdering(false);
	openingFilter3->SetFullyConnected(true);
	openingFilter3->SetAttribute("Elongation");
	openingFilter3->SetLambda(4);
	openingFilter3->Update();

	//label grains
	typedef itk::BinaryImageToShapeLabelMapFilter<UCharImageType> 
		BinaryToShapeFilterType;
	BinaryToShapeFilterType::Pointer binary2Shape = BinaryToShapeFilterType::New();
	binary2Shape->SetInput(openingFilter3->GetOutput());
	binary2Shape->SetComputeFeretDiameter(true);
	binary2Shape->Update();

	//label all grains
	BinaryToShapeFilterType::Pointer binary2ShapeAll = BinaryToShapeFilterType::New();
	binary2ShapeAll->SetInput(filter->GetOutput());
	binary2ShapeAll->SetComputeFeretDiameter(true);
	binary2ShapeAll->Update();

	float allGrains = binary2ShapeAll->GetOutput()->GetNumberOfLabelObjects(); 
	count = binary2Shape->GetOutput()->GetNumberOfLabelObjects(); 
	std::cout << "There are " << count << " shape label objects." << std::endl << std::endl;

	BinaryToShapeFilterType::OutputImageType::LabelObjectType* labelObject;

	std::vector<float> regionSize;
	for (i = 0; i < count; i++)
	{
		labelObject = binary2Shape->GetOutput()->GetNthLabelObject(i);
		regionSize.push_back(labelObject->GetPhysicalSize());
	}


	float sum = 0, mean, diff, sigma;

	for (i = 0; i < count; i++)
	{
		sum += regionSize[i];
	}

	mean = sum / count;
	float fraction = count/allGrains;
	sum = 0;
	std::cout << "fraction is " << fraction << std::endl;
	std::cout << "average is " << mean << std::endl;

	for (i = 0; i < count; i++)
	{
		sum += pow(regionSize[i] - mean, 2);
	}

	sigma = sqrt(sum / count);


	std::cout << "standard deviation is " << sigma << std::endl;

	typedef itk::LabelMapOverlayImageFilter<BinaryToShapeFilterType::OutputImageType, UCharImageType, RGBImageType> OverlayType;
	OverlayType::Pointer overlay = OverlayType::New();
	overlay->SetInput(binary2Shape->GetOutput());
	overlay->SetFeatureImage(featureReader->GetOutput());
	overlay->SetOpacity(1);
	overlay->Update();

	typedef itk::ImageFileWriter<RGBImageType> ColorWriterType;
	ColorWriterType::Pointer colorwriter = ColorWriterType::New();
	colorwriter->SetInput(overlay->GetOutput());
	colorwriter->SetFileName("riceOutput.png");
	colorwriter->Update();

	return 0;

}


