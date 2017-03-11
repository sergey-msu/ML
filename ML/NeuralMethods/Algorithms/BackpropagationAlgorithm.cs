using System;
using System.Collections.Generic;
using System.Linq;
using ML.Core;
using ML.NeuralMethods.Model;
using ML.Contracts;

namespace ML.NeuralMethods.Algorithms
{
  /// <summary>
  /// Multi-layer neural network training algorithm.
  /// Using Backpropagation principle as a core.
  /// </summary>
  public class BackpropagationAlgorithm : NeuralNetworkAlgorithmBase
  {
    #region Inner

    public enum StopCriteria
    {
      FullLoop = 0,
      ErrorFunc = 1,
      StepMin  = 2,
      QFunc    = 3
    }

    #endregion

    #region CONST

    public const int    DFT_EPOCH_COUNT = 1;
    public const double DFT_LEARNING_RATE = 0.1D;
    public const double DFT_Q_LAMBDA = 0.9D;
    public const StopCriteria DTF_STOP_CRITERIA = StopCriteria.FullLoop;
    public static readonly IFunction DFT_ACTIVATION_FUNCTION = Registry.ActivationFunctions.Identity;

    #endregion

    #region Fields

    private static double[][] m_BackpropErrors;

    private Dictionary<Class, double[]> m_ExpectedOutputs;
    private int m_EpochLength;

    private StopCriteria m_Stop;
    private int    m_InputDim;
    private int    m_OutputDim;
    private int    m_EpochCount;
    private double m_LearningRate;

    private double m_IterErrorValue;
    private double m_PrevErrorValue;
    private double m_ErrorValue;
    private double m_ErrorDelta;
    private double m_ErrorStopDelta;

    private double m_Step2;
    private double m_StepStopValue;

    private double m_QLambda;
    private double m_PrevQValue;
    private double m_QValue;
    private double m_QDelta;
    private double m_QStopDelta;

    #endregion

    #region .ctor

    public BackpropagationAlgorithm(ClassifiedSample classifiedSample, NeuralNetwork net)
      : base(classifiedSample, net)
    {
      init();
    }

    #endregion

    #region Events

    public event EventHandler EpochEndedEvent;

    #endregion

    #region Properties

    public override string ID { get { return "MLP_BP"; } }
    public override string Name { get { return "MLP Neural Network with Backpropagation"; } }

    public int InputDim           { get { return m_InputDim; } }
    public int OutputDim          { get { return m_OutputDim; } }
    public double IterErrorValue  { get { return m_IterErrorValue; } }
    public double ErrorValue      { get { return m_ErrorValue; } }
    public double ErrorDelta      { get { return m_ErrorDelta; } }
    public double Step2           { get { return m_Step2; } }
    public double QValue          { get { return m_QValue; } }
    public double QDelta          { get { return m_QDelta; } }

    public int EpochCount
    {
      get { return m_EpochCount; }
      set
      {
        if (value <= 0)
          throw new MLException("Epoch count must be positive");
        m_EpochCount = value;
      }
    }
    public double LearningRate
    {
      get { return m_LearningRate; }
      set
      {
        if (LearningRate <= 0)
          throw new MLException("Learning rate must be positive");
        m_LearningRate = value;
      }
    }
    public StopCriteria Stop
    {
      get { return m_Stop; }
      set { m_Stop = value; }
    }
    public double ErrorStopDelta
    {
      get { return m_ErrorStopDelta; }
      set
      {
        if (ErrorStopDelta < 0)
          throw new MLException("Error Stop Delta must be non-negative");
        m_ErrorStopDelta = value;
      }
    }
    public double StepStopValue
    {
      get { return m_StepStopValue; }
      set
      {
        if (StepStopValue < 0)
          throw new MLException("Step Stop Value must be non-negative");
        m_StepStopValue = value;
      }
    }
    public double QLambda
    {
      get { return m_QLambda; }
      set
      {
        if (QLambda < 0 || QLambda > 1)
          throw new MLException("Lambda for Q-stop criteria must be in [0, 1] interval");
        m_QLambda = value;
      }
    }
    public double QStopDelta
    {
      get { return m_QStopDelta; }
      set
      {
        if (QStopDelta < 0)
          throw new MLException("Q Stop Delta must be non-negative");
        m_QStopDelta = value;
      }
    }

    #endregion

    #region Public

    public void RunEpoch()
    {
      runEpoch(Result);
    }

    public void RunIteration(double[] data, Class cls)
    {
      runIteration(Result, data, cls);
    }

    #endregion

    public override void Train()
    {
      doTrain(Result);
    }

    #region .pvt

    private void init()
    {
      // apply defaults

      m_EpochCount   = DFT_EPOCH_COUNT;
      m_LearningRate = DFT_LEARNING_RATE;
      m_Stop         = DTF_STOP_CRITERIA;
      m_QLambda      = DFT_Q_LAMBDA;

      // parameters

      m_EpochLength  = TrainingSample.Count;

      var lcount = Result.LayerCount;
      m_InputDim = Result.InputDim;
      m_OutputDim = Result[lcount-1].NeuronCount;
      m_BackpropErrors = new double[lcount][];
      for (int i=0; i<lcount; i++)
        m_BackpropErrors[i] = new double[Result[i].NeuronCount];

      // expected outputs

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

    private void doTrain(NeuralNetwork net)
    {
      for (int epoch=0; epoch<m_EpochCount; epoch++)
      {
        runEpoch(net);
        if (checkStopCriteria()) break;
      }
    }

    private void runEpoch(NeuralNetwork net)
    {
      foreach (var pdata in TrainingSample)
      {
        runIteration(net, pdata.Key, pdata.Value);
      }

      // update epoch stats
      m_PrevErrorValue = m_ErrorValue;
      m_ErrorValue = m_IterErrorValue/TrainingSample.Count;
      m_ErrorDelta = m_ErrorValue-m_PrevErrorValue;
      m_IterErrorValue = 0.0D;

      if (EpochEndedEvent != null) EpochEndedEvent(this, EventArgs.Empty);
    }

    private void runIteration(NeuralNetwork net, double[] input, Class cls)
    {
      var lcount = net.LayerCount;
      var serr2 = 0.0D;
      var sstep2 = 0.0D;

      // forward calculation
      var output = net.Calculate(input);
      var expect = m_ExpectedOutputs[cls];
      var errors = m_BackpropErrors[lcount-1];
      for (int j=0; j<m_OutputDim; j++)
      {
        var ej = output[j] - expect[j];
        errors[j] = ej;
        serr2 += ej*ej;
      }

      // error backpropagation
      for (int i=lcount-1; i>=0; i--)
      {
        var layer  = net[i];
        var ncount = layer.NeuronCount;
        errors = m_BackpropErrors[i];

        var player  = (i>0) ? net[i-1] : null;
        var pcount  = (i>0) ? player.NeuronCount : m_InputDim;
        var perrors = (i>0) ? m_BackpropErrors[i-1] : null;
        if (perrors!=null) Array.Clear(perrors, 0, pcount);

        for (int j=0; j<ncount; j++)
        {
          // calculate current layer "errors"
          var neuron  = layer[j];
          var ej = errors[j];
          var gj = ej * neuron.Derivative;
          var dj = m_LearningRate * gj;

          for (int h=0; h<pcount; h++)
          {
            // save "errors" for future use
            if (i>0) perrors[h] += gj * neuron[h];

            // weights update
            var value = (i==0) ? input[h] : player[h].Value;
            var dwj = dj * value;
            neuron[h] -= dwj;
            sstep2 += dwj*dwj;
          }

          // bias weight update
          neuron.Bias -= dj;
          sstep2 += dj*dj;
        }
      }

      // update iter stats
      m_IterErrorValue += serr2/2;
      m_PrevQValue = m_QValue;
      m_QValue = (1-m_QLambda)*m_QValue + m_QLambda*serr2/2;
      m_QDelta = m_QValue-m_PrevQValue;
      m_Step2  = sstep2;
    }

    private bool checkStopCriteria()
    {
      switch (Stop)
      {
        case StopCriteria.FullLoop:  return false;
        case StopCriteria.ErrorFunc: return Math.Abs(m_ErrorDelta) < m_ErrorStopDelta;
        case StopCriteria.QFunc:     return Math.Abs(m_QDelta) < m_QStopDelta;
        case StopCriteria.StepMin:   return m_Step2 < StepStopValue;
        default: throw new MLException("Unknown stop citeria");
      }
    }

    #endregion
  }
}
