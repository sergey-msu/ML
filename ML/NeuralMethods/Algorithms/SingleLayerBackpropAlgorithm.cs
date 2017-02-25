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
    #region Inner

    public enum TrainingMode
    {
      Online = 0,
      Batch = 1
    }

    public enum StopCriteria
    {
      ErrorMin = 0,
      StepMin = 1
    }

    #endregion

    #region CONST

    public const bool DFT_USE_BIAS = true;
    public const int DFT_EPOCH_COUNT = 1;
    public const TrainingMode DTF_TRAINING_MODE = TrainingMode.Online;
    public const StopCriteria DTF_STOP_CRITERIA = StopCriteria.ErrorMin;
    public const double DFT_LEARNING_RATE = 0.1D;
    public const double DFT_ERROR_LEVEL = 0.0D;
    public const double DFT_STOP_STEP_LEVEL = 0.0D;
    public static readonly IFunction DFT_ACTIVATION_FUNCTION = Registry.ActivationFunctions.Identity;

    #endregion

    private Dictionary<Class, double[]> m_ExpectedOutputs;
    private double m_Error;
    private double m_Step;

    public SingleLayerBackpropAlgorithm(ClassifiedSample classifiedSample)
      : base(classifiedSample)
    {
      ActivationFunction = DFT_ACTIVATION_FUNCTION;
      UseBias            = DFT_USE_BIAS;
      EpochCount         = DFT_EPOCH_COUNT;
      Mode               = DTF_TRAINING_MODE;
      Stop               = DTF_STOP_CRITERIA;
      StopStepLevel      = DFT_STOP_STEP_LEVEL;
      LearningRate       = DFT_LEARNING_RATE;
      ErrorLevel         = DFT_ERROR_LEVEL;
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
    public StopCriteria Stop { get; set; }
    public double ErrorLevel { get; set; }
    public double StopStepLevel { get; set; }
    public double[] InitialWeights { get; set; }

    public double Error { get { return m_Error; } }
    public double Step { get { return m_Step; } }


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
        var cls = Classes.FirstOrDefault(p => (int)p.Value.Value == i+1).Value;
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

      for (int epoch=0; epoch<epochCount; epoch++)
      {
        runEpoch(net);
        if (checkStopCriteria()) break;
      }
    }

    private void runEpoch(NeuralNetwork net)
    {
      int epochLen = TrainingSample.Count;
      var terr2 = 0.0D;
      var tstep2 = 0.0D;

      foreach (var data in TrainingSample)
      {
        double ierr2;
        double istep2;
        runIter(net, data, out ierr2, out istep2);

        terr2 += ierr2;
        tstep2 = Math.Max(tstep2, istep2);
      }

      m_Error = terr2/epochLen;
      m_Step = tstep2;
    }

    private void runIter(NeuralNetwork net, KeyValuePair<Point, Class> data, out double ierr2, out double istep2)
    {
      var n = InputDim;
      var m = OutputDim;
      var rate = LearningRate;
      var layer = net[0];
      var serr2 = 0.0D;
      var sstep2 = 0.0D;

      // forward calculation
      var result = net.Calculate(data.Key);
      var expect = m_ExpectedOutputs[data.Value];

      // error backpropagation
      for (int j=0; j<m; j++)
      {
        var neuron = layer[j];
        var oj = neuron.Value;
        var ej = expect[j] - result[j];
        var dj = rate * ej * neuron.ActivationFunction.Derivative(oj);

        for (int i=0; i<n; i++)
        {
          var dwj = dj * data.Key[i];
          neuron[i] += dwj;
          sstep2 += dwj*dwj;
        }

        if (UseBias)
        {
          neuron[n] += dj;
          sstep2 += dj*dj;
        }

        serr2 += ej*ej;
      }

      ierr2 = serr2/2;
      istep2 = sstep2;
    }

    private bool checkStopCriteria()
    {
      if (Stop == StopCriteria.ErrorMin)
        return m_Error < ErrorLevel;

      if (Stop == StopCriteria.StepMin)
        return m_Step < StopStepLevel;

      return false;
    }

    #endregion
  }
}
