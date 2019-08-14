import java.util.ArrayList;
import java.util.Random;
import java.io.File;
import java.io.FileWriter;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.BufferedReader;
import java.io.IOException;

// A 1 dimensional array is treated as a column vector
// A 2 dimensional array is treated as a matrix of the form [row][column]

public class NeuralNet
{
	private int numLayers;
	private int[] layerSizes;
	private double[][][] weights;
	private double[][] biases;

	// construct neural net based on existing file
	public NeuralNet(String fileName)
	{
		loadFromFile(fileName);
	}

	// construct untrained neural net with given layer architecture
	public NeuralNet(int[] hiddenLayers)
	{
		numLayers = hiddenLayers.length;
		layerSizes = new int[numLayers];
		weights = new double[numLayers - 1][][];
		biases = new double[numLayers - 1][];
		
		// initialize layerSizes
		for(int i = 0; i < hiddenLayers.length; i++)
			layerSizes[i] = hiddenLayers[i];

		// create weight and bias layers
		for(int i = 0; i < numLayers - 1; i++)
		{
			weights[i] = new double[layerSizes[i + 1]][layerSizes[i]];
			biases[i] = new double[layerSizes[i + 1]];
		}

		// populate weights and biases randomly with gaussian distribution
		Random randNum = new Random();
		for(int i = 0; i < weights.length; i++)
		{
			for(int j = 0; j < weights[i].length; j++)
			{
				biases[i][j] = randNum.nextGaussian();
				for(int k = 0; k < weights[i][j].length; k++)
					weights[i][j][k] = randNum.nextGaussian();
			}
		}
	}

	public double[] feedforward(double[] input)
	{
		for(int i = 0; i < numLayers - 1; i++)
			input = sigmoid(matrixAdd(matrixMultiply(weights[i], input), biases[i]));	
			
		return input;
	}

	public void stochasticGradientDescent(double[][][] trainingData, double eta, int miniBatchSize, int epochs)
	{
		for(int e = 0; e < epochs; e++)
		{
			shuffle(trainingData);
			for(int i = 0; i < trainingData.length; i += miniBatchSize)
			{
				int size = i + miniBatchSize > trainingData.length ? trainingData.length - i : miniBatchSize; 
				double[][][] miniBatch = new double[size][][];
				for(int j = 0; j < miniBatch.length; j++)
					miniBatch[j] = trainingData[i + j];
				updateMiniBatch(miniBatch, eta); 
			}
			
			if(e % 10 == 0)
				System.out.println("Epoch " + e + " finished");
		}
	}	

	public void updateMiniBatch(double[][][] miniBatch, double eta)
	{
		// initialize weightDelta and biasDelta
		double[][][] weightDelta = new double[numLayers - 1][][];
		double[][] biasDelta = new double[numLayers - 1][];

		for(int i = 0; i < numLayers - 1; i++)
		{
			weightDelta[i] = new double[layerSizes[i + 1]][layerSizes[i]];
			biasDelta[i] = new double[layerSizes[i + 1]];

			for(int j = 0; j < weightDelta[i].length; j++)
			{
				biasDelta[i][j] = 0;
				for(int k = 0; k < weightDelta[i][j].length; k++)
					weightDelta[i][j][k] = 0;
			}
		}

		// update weights and biases
		for(int i = 0; i < miniBatch.length; i++)
		{
			double[][][] weightPartials = new double[numLayers - 1][][];
			double[][] biasPartials = new double[numLayers - 1][];
			backpropagation(miniBatch[i][0], miniBatch[i][1], weightPartials, biasPartials);
			
			for(int a = 0; a < weightDelta.length; a++)
			{
				weightDelta[a] = matrixAdd(weightDelta[a], weightPartials[a]);
				biasDelta[a] = matrixAdd(biasDelta[a], biasPartials[a]);	
			}
		}

		for(int i = 0; i < weights.length; i++)
		{
			for(int j = 0; j < weights[i].length; j++)
			{
				biases[i][j] -= (eta / miniBatch.length) * biasDelta[i][j];	
				for(int k = 0; k < weights[i][j].length; k++)
					weights[i][j][k] -= (eta / miniBatch.length) * weightDelta[i][j][k];
			}
		}
	}

	public void backpropagation(double[] trainingInput, double[] desiredOutput, double[][][] weightPartials, double[][] biasPartials)
	{
		double[][] activations = new double[numLayers][];
		double[][] ZValues = new double[numLayers - 1][];

		// forward propagate
		double[] activation = trainingInput;
		activations[0] = activation;	
		for(int layer = 0; layer < numLayers - 1; layer++)
		{
			activation = matrixAdd(matrixMultiply(weights[layer], activation), biases[layer]);
			ZValues[layer] = new double[activation.length];
			for(int i = 0; i < activation.length; i++)
				ZValues[layer][i] = activation[i];

			activation = sigmoid(activation);
			activations[layer + 1] = new double[activation.length];
			for(int i = 0; i < activation.length; i++)
				activations[layer + 1][i] = activation[i];
		}	

		// backward propagate
		double[][] error = new double[numLayers - 1][];
		
		double[] costDeriv = costDerivative(activations[activations.length - 1], desiredOutput);
		double[] ZDeriv = sigmoidPrime(ZValues[ZValues.length - 1]);
		error[error.length - 1] = hadamardProduct(costDeriv, ZDeriv);

		for(int layer = numLayers - 2; layer > 0; layer--)
		{
			double[] matrixProduct = matrixMultiply(transpose(weights[layer]), error[layer]);
			error[layer - 1] = hadamardProduct(matrixProduct, sigmoidPrime(ZValues[layer - 1]));
		}

		for(int i = 0; i < error.length; i++)
		{
			biasPartials[i] = error[i];
			weightPartials[i] = matrixMultiply(error[i], transpose(activations[i]));
		}
	}

	public double sigmoid(double z)
	{
		return 1 / (1 + Math.pow(Math.E, z * -1));
	}

	public double[] sigmoid(double[] z)
	{
		double[] result = new double[z.length];
		for(int i = 0; i < z.length; i++)
			result[i] = sigmoid(z[i]);
	
		return result;
	}

	public double sigmoidPrime(double z)
	{
		return sigmoid(z) * (1 - sigmoid(z));
	}

	public double[] sigmoidPrime(double[] z)
	{
		double[] result = new double[z.length];
		for(int i = 0; i < z.length; i++)
			result[i] = sigmoidPrime(z[i]);

		return result;
	}

	public double[] costDerivative(double[] outputActivations, double[] desiredActivations)
	{
		double[] result = new double[outputActivations.length];
		for(int i = 0; i < outputActivations.length; i++)
			result[i] = outputActivations[i] - desiredActivations[i];
	
		return result;
	}

	public void shuffle(double[][][] arr)
	{
		Random randNum = new Random();
		for(int i = arr.length - 1; i > 0; i--)
		{
			int swapIndex = randNum.nextInt(i);
			double[][] temp = arr[swapIndex];
			arr[swapIndex] = arr[i];
			arr[i] = temp;
		}
	}

	// transpose of a 1 dimensional array
	public double[][] transpose(double[] matrix)		
	{
		double[][] result = new double[1][matrix.length];
		for(int i = 0; i < matrix.length; i++)
			result[0][i] = matrix[i];

		return result;
	}

	// transpose of a 2 dimensional array
	public double[][] transpose(double[][] matrix)
	{
		double[][] result = new double[matrix[0].length][matrix.length];
		for(int row = 0; row < matrix.length; row++)
		{
			for(int col = 0; col < matrix[row].length; col++)
				result[col][row] = matrix[row][col];
		}

		return result;
	}

	// multiply a matrix by a column vector
	public double[] matrixMultiply(double[][] first, double[] second)
	{
		double[] result = new double[first.length];
		for(int row = 0; row < first.length; row++)
		{
			double sum = 0;
			for(int col = 0; col < first[row].length; col++)
				sum += first[row][col] * second[col];
		
			result[row] = sum;
		}

		return result;
	}

	// multiply a column vector by a row vector
	public double[][] matrixMultiply(double[] first, double[][] second)
	{
		double[][] result = new double[first.length][second[0].length];
		for(int row = 0; row < result.length; row++)
		{
			for(int col = 0; col < result[row].length; col++)
				result[row][col] = first[row] * second[0][col];
		}

		return result;
	}

	// add two column vectors elementwise
	public double[] matrixAdd(double[] first, double[] second)
	{
		double[] result = new double[first.length];
		for(int i = 0; i < first.length; i++)
			result[i] = first[i] + second[i];
	
		return result;
	}

	// add two matrices elementwise
	public double[][] matrixAdd(double[][] first, double[][] second)
	{
		double[][] result = new double[first.length][first[0].length];
		for(int row = 0; row < first.length; row++)
		{
			for(int col = 0; col < first[0].length; col++)
				result[row][col] = first[row][col] + second[row][col];
		}

		return result;
	}

	// multiply two column vectors elementwise
	public double[] hadamardProduct(double[] first, double[] second)
	{
		double[] result = new double[first.length];
		for(int i = 0; i < first.length; i++)
			result[i] = first[i] * second[i];

		return result;
	}

	public void printLayers()
	{
		System.out.println("Number of layers: " + numLayers);
		for(int i = 0; i < layerSizes.length; i++)
			System.out.print(layerSizes[i] + " ");
		
		System.out.println();
		for(int i = 0; i < weights.length; i++)
		{
			System.out.println("Layer: " + i);
			for(int j = 0; j < weights[i].length; j++)
			{
				for(int k = 0; k < weights[i][j].length; k++)
					System.out.print(weights[i][j][k] + " ");
			
				System.out.println();
			}
		}

		System.out.println();
		System.out.println("Biases: ");
		for(int i = 0; i < biases.length; i++)
		{
			System.out.println("Layer: " + i);
			for(int j = 0; j < biases[i].length; j++)
				System.out.print(biases[i][j] + " ");

			System.out.println();
		}

		System.out.println();
	}

	public void recordToFile(String fileName)
	{
		File file = new File(fileName);
		try
		{
			if(!file.createNewFile())
			{
				file.delete();
				file.createNewFile();
			}

			BufferedWriter writer = new BufferedWriter(new FileWriter(file));
			
			// write layers
			writer.write("Layers");
			writer.newLine();
			for(int i = 0; i < layerSizes.length; i++)
			{
				writer.write(Integer.toString(layerSizes[i]));
				writer.newLine();
			}
			writer.newLine();

			// write weights
			writer.write("Weights");
			writer.newLine();	
			for(int i = 0; i < weights.length; i++)
			{	
				writer.write("Layer " + Integer.toString(i));
				writer.newLine();
				for(int j = 0; j < weights[i].length; j++)
				{
					for(int k = 0; k < weights[i][j].length; k++)
					{
						writer.write(Double.toString(weights[i][j][k]));
						writer.newLine();
					}
					writer.newLine();
						
				}
				writer.newLine();
			}
	
			writer.write("Biases");
			writer.newLine();
			for(int i = 0; i < biases.length; i++)
			{
				writer.write("Layer " + Integer.toString(i));
				writer.newLine();
				for(int j = 0;j < biases[i].length; j++)
				{
					writer.write(Double.toString(biases[i][j]));
					writer.newLine();
				}
				writer.newLine();
			}
				
			writer.write("END");
			writer.flush();
			writer.close();
		}
		catch(IOException e)
		{
			System.out.println("recordToFile IOException");
		}
	}

	public void loadFromFile(String fileName)
	{	
		final String NEW_LINE = System.getProperty("line.separator");
		File file = new File(fileName);
		try
		{
			BufferedReader reader = new BufferedReader(new FileReader(file));
			int mode = 0;
			
			for(;;)
			{
				String nextLine = reader.readLine();
			
				if(nextLine.equals("END"))
					break;
				else if(nextLine.equals("Layers"))
					mode = 1;
				else if(nextLine.equals("Weights"))
					mode = 2;
				else if(nextLine.equals("Biases"))
					mode = 3;

				switch(mode)
				{
				// layerSizes
				case 1:
					ArrayList<Integer> sizes = new ArrayList<Integer>();

					nextLine = reader.readLine();
					while(!nextLine.equals(""))
					{
						sizes.add(Integer.parseInt(nextLine));
						nextLine = reader.readLine();
					}
				
					// initialize layerSizes
					numLayers = sizes.size();
					layerSizes = new int[numLayers];
					for(int i = 0; i < sizes.size(); i++)
						layerSizes[i] = sizes.get(i);

					// create weights and biases arrays
					weights = new double[numLayers - 1][][];
					for(int i = 0; i < weights.length; i++)
						weights[i] = new double[layerSizes[i + 1]][layerSizes[i]];
				
					biases = new double[numLayers - 1][];
					for(int i = 0; i < biases.length; i++)
						biases[i] = new double[layerSizes[i + 1]];
					
					break;
				// weights
				case 2:
					int weightLayerCount = 0;
					while(weightLayerCount < weights.length)
					{
						nextLine = reader.readLine();
						nextLine = reader.readLine();

						int row = 0; 
						while(!nextLine.equals(""))
						{
							int col = 0;
							while(!nextLine.equals(""))
							{
								weights[weightLayerCount][row][col] = Double.parseDouble(nextLine);
								col++;
								nextLine = reader.readLine();
							}
							
							row++;	
							nextLine = reader.readLine();
						}

						weightLayerCount++;
					}
					break;
				// biases
				case 3:
					int biasLayerCount = 0;
					while(biasLayerCount < biases.length)
					{
						nextLine = reader.readLine();
						nextLine = reader.readLine();

						int index = 0;
						while(!nextLine.equals(""))
						{
							biases[biasLayerCount][index] = Double.parseDouble(nextLine);
							index++;
							nextLine = reader.readLine();
						}  

						biasLayerCount++;
					}
					break;
				default:
					continue;
				}
			}
		}
		catch(IOException e)
		{
			System.out.println("loadFromFile IOException");
		}
	}
}