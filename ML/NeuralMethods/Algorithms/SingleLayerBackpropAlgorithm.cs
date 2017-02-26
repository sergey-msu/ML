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
    public const double DFT_ERROR_DELTA = 0.0D;
    public const double DFT_STOP_STEP_LEVEL = 0.0D;
    public static readonly IFunction DFT_ACTIVATION_FUNCTION = Registry.ActivationFunctions.Identity;

    #endregion

    private Dictionary<Class, double[]> m_ExpectedOutputs;
    private double       m_PrevError;
    private double       m_Error;
    private double       m_Step;
    private IFunction    m_ActivationFunction;
    private int          m_InputDim;
    private int          m_OutputDim;
    private bool         m_UseBias;
    private int          m_EpochCount;
    private double       m_LearningRate;
    private TrainingMode m_Mode;
    private int          m_BatchSize;
    private StopCriteria m_Stop;
    private double       m_ErrorDelta;
    private double       m_StopStepLevel;
    private double[]     m_InitialWeights;

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
      ErrorDelta         = DFT_ERROR_DELTA;
    }

    #region Properties

    public override string ID { get { return "SL_BP"; } }
    public override string Name { get { return "Single Layer Percentron with Backpropagation"; } }

    public IFunction ActivationFunction
    {
      get { return m_ActivationFunction; }
      set { m_ActivationFunction = value; }
    }
    public int InputDim
    {
      get { return m_InputDim; }
      set { m_InputDim = value; }
    }
    public int OutputDim
    {
      get { return m_OutputDim; }
      set { m_OutputDim = value; }
    }
    public bool UseBias
    {
      get { return m_UseBias; }
      set { m_UseBias = value; }
    }
    public int EpochCount
    {
      get { return m_EpochCount; }
      set { m_EpochCount = value; }
    }
    public double LearningRate
    {
      get { return m_LearningRate; }
      set { m_LearningRate = value; }
    }
    public TrainingMode Mode
    {
      get { return m_Mode; }
      set { m_Mode = value; }
    }
    public int BatchSize
    {
      get { return m_BatchSize; }
      set { m_BatchSize = value; }
    }
    public StopCriteria Stop
    {
      get { return m_Stop; }
      set { m_Stop = value; }
    }
    public double ErrorDelta
    {
      get { return m_ErrorDelta; }
      set { m_ErrorDelta = value; }
    }
    public double StopStepLevel
    {
      get { return m_StopStepLevel; }
      set { m_StopStepLevel = value; }
    }
    public double[] InitialWeights
    {
      get { return m_InitialWeights; }
      set { m_InitialWeights = value; }
    }

    public double Error { get { return m_Error; } }
    public double Step { get { return m_Step; } }

    #endregion

    protected override NeuralNetwork DoTrain()
    {
      check();
      prepareExpectedOutputs();
      var net = constructNetwork();

      switch (Mode)
      {
        case TrainingMode.Online: doOnlineTrain(net); break;
        case TrainingMode.Batch:  doBatchTrain(net); break;
        default: throw new MLException("Unknown training mode");
      }

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
      if (ErrorDelta < 0) throw new MLException("Error level must be non-negative");

      var pcount = UseBias ? (InputDim+1)*OutputDim : InputDim*OutputDim;
      if (InitialWeights!=null && InitialWeights.Length != pcount)
        throw new MLException(string.Format("The algorithm assumes full-connectivity, therefore parameter count must be inputs*outputs={0}*{1}={2}", InputDim, OutputDim, pcount));

      if (Mode==TrainingMode.Batch && BatchSize<=0)
        BatchSize = TrainingSample.Count;
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

    private bool checkStopCriteria()
    {
      switch (Stop)
      {
        case StopCriteria.ErrorMin: return Math.Abs(m_Error-m_PrevError) < ErrorDelta;
        case StopCriteria.StepMin:  return m_Step < StopStepLevel;
        default: throw new MLException("Unknown stop citeria");
      }
    }

    #region Online Training

    private void doOnlineTrain(NeuralNetwork net)
    {
      for (int epoch=0; epoch<m_EpochCount; epoch++)
      {
        runEpoch(net);
        if (checkStopCriteria()) break;
      }
    }

    private void runEpoch(NeuralNetwork net)
    {
      int l = TrainingSample.Count;
      double ierr2;
      double terr2 = 0.0D;
      double istep2;
      double tstep2 = 0.0D;

      foreach (var pdata in TrainingSample)
      {
        runOnlineIter(net, pdata.Key, pdata.Value, out ierr2, out istep2);

        terr2 += ierr2;
        tstep2 = Math.Max(tstep2, istep2);
      }

      m_PrevError = m_Error;
      m_Error = terr2/l;
      m_Step = tstep2;
    }

    private void runOnlineIter(NeuralNetwork net,
                               Point data, Class cls,
                               out double ierr2, out double istep2)
    {
      var layer = net[0];
      var serr2 = 0.0D;
      var sstep2 = 0.0D;

      // forward calculation
      var result = net.Calculate(data);
      var expect = m_ExpectedOutputs[cls];

      // error backpropagation
      for (int j=0; j<m_OutputDim; j++)
      {
        var neuron = layer[j];
        var oj = neuron.Value;
        var ej = expect[j] - result[j];
        var dj = m_LearningRate * ej * neuron.ActivationFunction.Derivative(oj);

        for (int i=0; i<m_InputDim; i++)
        {
          var dwj = dj * data[i];
          neuron[i] += dwj;
          sstep2 += dwj*dwj;
        }

        if (UseBias)
        {
          neuron[m_InputDim] += dj;
          sstep2 += dj*dj;
        }

        serr2 += ej*ej;
      }

      ierr2 = serr2/2;
      istep2 = sstep2;
    }

    #endregion

    #region Batch Training

    private void doBatchTrain(NeuralNetwork net)
    {
      for (int epoch=0; epoch<m_EpochCount; epoch++)
      {
        runBatchEpoch(net);
        if (checkStopCriteria()) break;
      }
    }

    private void runBatchEpoch(NeuralNetwork net)
    {
      var l = TrainingSample.Count;
      var pcount = m_UseBias ? (m_InputDim+1)*m_OutputDim : m_InputDim*m_OutputDim;
      var deltas = new double[pcount];
      double ierr2;
      double terr2 = 0.0D;
      double istep2;
      double tstep2 = 0.0D;
      var idx = 0;
      var bcount = 0;

      foreach (var pdata in TrainingSample)
      {
        runBatchIter(net, pdata.Key, pdata.Value, deltas, out ierr2);

        if (bcount>=m_BatchSize || idx>=l-1)
        {
          updateWeights(net, deltas, out istep2);
          tstep2 = Math.Max(tstep2, istep2);
          bcount = -1;
        }

        terr2 += ierr2;
        bcount++;
        idx++;
      }

      m_PrevError = m_Error;
      m_Error = terr2/l;
      m_Step = tstep2;
    }

    private void runBatchIter(NeuralNetwork net,
                              Point data, Class cls,
                              double[] deltas, out double ierr2)
    {
      var layer = net[0];
      var serr2 = 0.0D;
      var pidx = 0;

      // forward calculation
      var result = net.Calculate(data);
      var expect = m_ExpectedOutputs[cls];

      // error backpropagation & weights update
      for (int j=0; j<m_OutputDim; j++)
      {
        var neuron = layer[j];
        var oj = neuron.Value;
        var ej = expect[j] - result[j];
        var dj = m_LearningRate * ej * neuron.ActivationFunction.Derivative(oj);

        for (int i=0; i<m_InputDim; i++)
        {
          var dwj = dj * data[i];
          deltas[pidx++] += dwj;
        }

        if (m_UseBias)
          deltas[pidx++] += dj;

        serr2 += ej*ej;
      }

      ierr2 = serr2/2;
    }

    public void updateWeights(NeuralNetwork net, double[] deltas, out double istep2)
    {
      var layer = net[0];
      var sstep2 = 0.0D;
      var pidx = 0;

      for (int j=0; j<m_OutputDim; j++)
      {
        var neuron = layer[j];

        for (int i=0; i<m_InputDim; i++)
        {
          var dwj = deltas[pidx];
          deltas[pidx++] = 0;

          neuron[i] += dwj;
          sstep2 += dwj*dwj;
        }

        if (m_UseBias)
        {
          var dj = deltas[pidx];
          deltas[pidx++] = 0;

          neuron[m_InputDim] += dj;
          sstep2 += dj*dj;
        }

        deltas[j] = 0;
      }

      istep2 = sstep2;
    }

    #endregion

    #endregion
  }
}
