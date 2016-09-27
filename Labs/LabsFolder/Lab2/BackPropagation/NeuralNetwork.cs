using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    #region Функції активації та їхні похідні
    // enum для перелічення можливих функцій активації
    public enum TransferFunction
    {
        None,
        Sigmoid,
        BipolarSigmoid
    } 
    
    public static class TransferFunctions
    {
        // Функція активації
        public static double Evaluate(TransferFunction tFunc, double input)
        {
            switch (tFunc)
            {
                case TransferFunction.Sigmoid:
                    return sigmoid(input);
                case TransferFunction.BipolarSigmoid:
                    return bipolarsigmoid(input);
                case TransferFunction.None :
                default:
                    return 0.0;
            }
        }

        // Похідна функції активації
        public static double DerivativeEvaluate(TransferFunction tFunc, double input)
        {
            switch (tFunc)
            {
                case TransferFunction.Sigmoid:
                    return sigmoid_derivative(input);
                case TransferFunction.BipolarSigmoid:
                    return bipolarsigmoid_derivative(input);
                case TransferFunction.None :
                default:
                    return 0.0;

            }

        }

        // Сигмоїда
        private static double sigmoid(double x)
        {
            return 1.0 / (1.0 + Math.Exp(-x));
        }

        // Похідна сигмоїди
        private static double sigmoid_derivative(double x)
        {
            return sigmoid(x) * (1 - sigmoid(x));
        }

        private static double bipolarsigmoid(double x)
        {
            return 2.0 / (1.0 + Math.Exp(-x)) - 1;
        }

        private static double bipolarsigmoid_derivative(double x)
        {
            return 0.5 * (1 + bipolarsigmoid(x)) * (1 - bipolarsigmoid(x));
        }

    }
    #endregion

    public class BackPropagationNetwork
    {
        #region Поля
        private int layerCount; // прихований шар + вихідний шар
        private int inputSize; // к-сть нейронів у вхдному шарі
        private int[] layerSize; // величини к-сті нейронів у прихованому та вихідному шарах 
        private TransferFunction[] transferFunction; // масив функцій активації

        private double[][] layerOtput; // вихідні дані шару
        private double[][] layerInput; // вхідні дані шару
        private double[][] bias; // відхилення
        private double[][] delta; // дельта помилки
        private double[][] previosBiasDelta; // дельта попереднього відхилення

        private double[][][] weight; // ваги, де [0] - шар, [1] - попередній нейрон, [2] - поточний нейрон
        private double[][][] previousWeightDelta; // дельта попередньої ваги


        #endregion

        #region Конструктор
        public BackPropagationNetwork(int[] layerSizes, TransferFunction[] TransferFunctions)
        {
            // Перевірка вхідних даних
            if (TransferFunctions.Length != layerSizes.Length || TransferFunctions[0] != TransferFunction.None)
                throw new ArgumentException("The network cannot be created with these parameters");
            
            // Ініціалізація шарів мережі
            layerCount = layerSizes.Length - 1;
            inputSize = layerSizes[0];
            layerSize = new int[layerCount];
            transferFunction = new TransferFunction[layerCount];

            for (int i = 0; i<layerCount; i++) 
                layerSize[i] = layerSizes[i + 1];

            for (int i = 0; i < layerCount; i++)
                transferFunction[i] = TransferFunctions[i + 1];

            // Визначення вимірів масивів
            bias = new double[layerCount][];
            previosBiasDelta = new double[layerCount][];
            delta = new double[layerCount][];
            layerOtput = new double[layerCount][];
            layerInput = new double[layerCount][];

            weight = new double[layerCount][][];
            previousWeightDelta = new double[layerCount][][];

            // Заповнення двовимірних масивів
            for (int l = 0; l<layerCount; l++)
            {
                bias[l] = new double[layerSize[l]];
                previosBiasDelta[l] = new double[layerSize[l]];
                delta[l] = new double[layerSize[l]];
                layerOtput[l] = new double[layerSize[l]];
                layerInput[l] = new double[layerSize[l]];

                weight[l] = new double[l == 0 ? inputSize : layerSize[l-1]][];
                previousWeightDelta[l] = new double[l == 0 ? inputSize : layerSize[l-1]][];

                for (int i = 0; i<(l == 0 ? inputSize : layerSize[l - 1]); i++)
                {
                    weight[l][i] = new double[layerSize[l]];
                    previousWeightDelta[l][i] = new double[layerSize[l]];
                }
            }

            // Ініціалізація ваг
            for(int l =0; l < layerCount; l++)
            {
                for(int i = 0; i < layerSize[l]; i++)
                {
                    bias[l][i] = Gaussian.GetRandomGaussian();
                    previosBiasDelta[l][i] = 0.0;
                    layerInput[l][i] = 0.0;
                    layerOtput[l][i] = 0.0;
                    delta[l][i] = 0.0;
                }

                for(int i = 0; i< (l == 0 ? inputSize : layerSize[l - 1]); i++)
                {
                    for (int j = 0; j < layerSize[l]; j++) {
                        weight[l][i][j] = Gaussian.GetRandomGaussian();
                        previousWeightDelta[l][i][j] = 0.0;
                    }
                }
            }
        }

        #endregion

        #region Methods
        public void Run(ref double[] input, out double[] output)
        {
            // Перевірка, чи введені дані відповідають кількості нейронів у вхідному шарі
            if (input.Length != inputSize)
                throw new ArgumentException("Input data isn't of the correct dimension");

            // Вихідне значення функції
            output = new double[layerSize[layerCount-1]];

            // Нормалізація вхідних значень
            double max = input.Max();
           
            // Запуск мережі
            for(int l = 0; l<layerCount; l++)
            {
                for(int j = 0; j < layerSize[l]; j++)
                {
                    double sum = 0.0;
                    for(int i = 0; i<(l == 0 ? inputSize : layerSize[l-1]); i++)
                        sum += weight[l][i][j] * (l == 0 ? input[i] : layerOtput[l-1][i]);

                    sum += bias[l][j];
                    layerInput[l][j] = sum;

                    layerOtput[l][j] = TransferFunctions.Evaluate(transferFunction[l], sum);   
           
                }
            }

            // копіюємо вихід мережі у вихідний масив
            for(int i = 0; i < layerSize[layerCount-1]; i++)
            {
                output[i] = layerOtput[layerCount - 1][i];
            }

        }

        // Функція навчання
        public double Train(ref double[] input, ref double[] desired, double TrainingRate, double Momentum)
        {
            // Перевірка вхідних параметрів
            if (input.Length != inputSize)
                throw new ArgumentException("Invalid input parameter", "input");

            if (desired.Length != layerSize[layerCount - 1])
                throw new ArgumentException("Invalid input parameter", "desired");

            // Локальні змінні
            double error = 0.0, sum = 0.0, weigtdelta = 0.0, biasDelta = 0.0;
            double[] output = new double[layerSize[layerCount-1]];

            // Запуск мережі
            Run(ref input, out output);

            //Розмножуємо похибку у зворотньму порядку
            for (int l = layerCount - 1; l>=0; l--)
            {
                //Вихідний шар
                if(l == layerCount - 1)
                {
                    for (int k = 0; k < layerSize[l]; k++)
                    {
                        delta[l][k] = output[k] - desired[k];
                        error += Math.Pow(delta[l][k], 2);
                        delta[l][k] *= TransferFunctions.DerivativeEvaluate(transferFunction[l], layerInput[l][k]);
                    }
                   
                }
                //Прихований шар
                else
                {
                    for (int i =0; i<layerSize[l]; i++)
                    {
                        sum = 0.0;
                        for (int j = 0; j < layerSize[l+1]; j++)
                        {
                            sum += weight[l + 1][i][j] * delta[l+1][j];
                        }
                        sum *= TransferFunctions.DerivativeEvaluate(transferFunction[l], layerInput[l][i]);
                        delta[l][i] = sum;
                    }
                }
            }

            // Оновлення ваг та відхилень
            for (int l = 0; l<layerCount; l++)
                for (int i = 0; i < (l == 0 ? inputSize : layerSize[l-1]); i++)
                    for (int j = 0; j < layerSize[l]; j++)
                    {
                        weigtdelta = TrainingRate * delta[l][j] * (l == 0 ? input[i] : layerOtput[l-1][i]);
                        weight[l][i][j] -= weigtdelta + Momentum * previousWeightDelta[l][i][j];

                        previousWeightDelta[l][i][j] = weigtdelta;
                    }

            for(int l =0; l < layerCount; l++)
                for(int i = 0; i < layerSize[l]; i++)
                {
                    biasDelta =-1 * TrainingRate * delta[l][i];
                    bias[l][i] += biasDelta + Momentum * previosBiasDelta[l][i];

                    previosBiasDelta[l][i] = biasDelta;
                }

            return error;     
        }
        #endregion
    }

    // Клас для створення випадкових чисел з нормальним розподілом
    public static class Gaussian
    {
        private static Random gen = new Random();

        public static double GetRandomGaussian()
        {
            return GetRandomGaussian(0.0, 1.0);
        }

        public static double GetRandomGaussian(double mean, double stddev)
        {
            double rVal1, rVal2;
            GetRandomGaussian(mean, stddev, out rVal1, out rVal2);
            return rVal1;
        }

        public static void GetRandomGaussian(double mean, double stddev, out double val1, out double val2)
        {
            double u, v, s, t;

            do
            {
                u = 2 * gen.NextDouble() - 1;
                v = 2 * gen.NextDouble() - 1;
            } while (u * u + v * v > 1 || (u == 0 && v == 0));

            s = u * u + v * v;
            t = Math.Sqrt((-2.0 * Math.Log(s)) / s);

            val1 = stddev * t * u + mean;
            val2 = stddev * t * v + mean;


        }
    }
}
