using System;
using System.Linq;
using System.Collections.Generic;
using ML.NeuralMethods.Model;
using ML.Core;
using ML.Contracts;

namespace ML.NeuralMethods.Algorithms
{
  /// <summary>
  /// Trains network with no hidden layer (only input and output myltivalued layer) via backpropagation algorithm
  /// </summary>
  public class SingleLayerBackpropAlgorithm : NeuralNetworkAlgorithmBase
  {
    #region CONST

    public const bool DFT_USE_BIAS = true;
    public const int DFT_EPOCH_COUNT = 1;
    public const TrainingMode DTF_TRAINING_MODE = TrainingMode.Sequential;
    public const double DFT_LEARNING_RATE = 0.1D;
    public const double DFT_ERROR_LEVEL = 0.0D;
    public static readonly IFunction DFT_ACTIVATION_FUNCTION = Registry.ActivationFunctions.Identity;

    #endregion

    #region Inner

    public enum TrainingMode
    {
      Sequential = 0,
      Batch = 1
    }

    #endregion

    private Dictionary<Class, double[]> m_ExpectedOutputs;
    private double m_Error;

    protected SingleLayerBackpropAlgorithm(ClassifiedSample classifiedSample)
      : base(classifiedSample)
    {
      UseBias = DFT_USE_BIAS;
      ActivationFunction = DFT_ACTIVATION_FUNCTION;
      EpochCount = DFT_EPOCH_COUNT;
      Mode = DTF_TRAINING_MODE;
      LearningRate = DFT_LEARNING_RATE;
      ErrorLevel = DFT_ERROR_LEVEL;
    }

    public override string ID { get { return "SL_BP"; } }
    public override string Name { get { return "Single Layer Percentron with Backpropagation"; } }

    public IFunction ActivationFunction { get; set; }
    public int InputDim { get; set; }
    public int OutputDim { get; set; }
    public bool UseBias { get; set; }
    public int EpochCount { get; set; }
    public double LearningRate { get ;set; }
    public TrainingMode Mode { get; set; }
    public double ErrorLevel { get; set; }
    public double[] InitialWeights { get; set; }

    public double Error { get { return m_Error; } }


    protected override NeuralNetwork DoTrain()
    {
      check();
      prepareExpectedOutputs();
      var net = constructNetwork();
      doTrain(net);

      return net;
    }

    #region .pvt

    private void check()
    {
      if (ActivationFunction==null) throw new MLException("Activaltion function is null");
      if (InputDim <= 0) throw new MLException("Input dimension must be positive");
      if (OutputDim <= 0) throw new MLException("Output dimension nust be positive");
      if (EpochCount <= 0) throw new MLException("Epoch count must be positive");
      if (LearningRate <= 0) throw new MLException("Learning rate must be positive");
      if (ErrorLevel < 0) throw new MLException("Error level must be non-negative");

      var pcount = InputDim*OutputDim;
      if (InitialWeights!=null && InitialWeights.Length != pcount)
        throw new MLException(string.Format("The algorithm assumes full-connectivity, therefore parameter count must be inputs*outputs={0}*{1}={2}", InputDim, OutputDim, pcount));
    }

    private void prepareExpectedOutputs()
    {
      m_ExpectedOutputs = new Dictionary<Class, double[]>();
      var count = Classes.Count;
      if (count != OutputDim)
        throw new MLException("Number of classes must be equal to dimetsion of output vector");

      for (int i=0; i<count; i++)
      {
        var cls = Classes.FirstOrDefault(p => (int)p.Value.Value == i).Value;
        if (cls==null)
          throw new MLException(string.Format("There is no class with value {0}. It is neccessary to have full set of classes with values from 0 to {1}", i, count));

        var output = new double[count];
        output[i] = 1.0D;
        m_ExpectedOutputs[cls] = output;
      }
    }

    private NeuralNetwork constructNetwork()
    {
      var net = new NeuralNetwork();
      net.InputDim = InputDim;
      net.UseBias = UseBias;
      net.ActivationFunction = ActivationFunction;

      var layer = net.CreateLayer();
      for (int i=0; i<OutputDim; i++)
        layer.CreateNeuron<FullNeuron>();

      if (InitialWeights != null)
      {
        int cursor = 0;
        net.TryUpdateParams(InitialWeights, false, ref cursor);
      }

      net.Build();

      return net;
    }

    private void doTrain(NeuralNetwork net)
    {
      int epochCount = EpochCount;
      var terr2 = 0.0D;

      for (int epoch=0; epoch<epochCount; epoch++)
      {
        terr2 = runEpoch(net);
        if (terr2 < ErrorLevel) break;
      }

      m_Error = terr2;
    }

    private double runEpoch(NeuralNetwork net)
    {
      int epochLen = TrainingSample.Count;
      var terr2 = 0.0D;

      foreach (var data in TrainingSample)
      {
        terr2 += runIter(net, data);
      }

      return terr2/epochLen;
    }

    private double runIter(NeuralNetwork net, KeyValuePair<Point, Class> data)
    {
      var n = InputDim;
      var m = OutputDim;
      var rate = LearningRate;
      var activation = ActivationFunction;
      var layer = net[0];
      var serr2 = 0.0D;

      // forward calculation
      var result = net.Calculate(data.Key);
      var expect = m_ExpectedOutputs[data.Value];

      // error backpropagation
      for (int j=0; j<m; j++)
      {
        var neuron = layer[j];
        var oj = neuron.Value;
        var ej = expect[j] - result[j];
        var dj = rate * ej * activation.Derivative(oj);

        for (int i=0; i<n; i++)
          neuron[i] += dj * data.Key[i];

        if (UseBias) neuron[n] += dj;

        serr2 += ej*ej;
      }

      return serr2/2;
    }

    #endregion
  }
}
