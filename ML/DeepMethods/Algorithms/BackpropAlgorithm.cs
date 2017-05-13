using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using ML.Core;
using ML.DeepMethods.Models;
using ML.Contracts;

namespace ML.DeepMethods.Algorithms
{
  /// <summary>
  /// Convolutional neural network training algorithm.
  /// Uses Backpropagation principle as a core.
  /// </summary>
  public partial class BackpropAlgorithm : ConvNetAlgorithmBase
  {
    #region Inner

    public enum StopCriteria
    {
      FullLoop = 0,
      ErrorFunc = 1,
      StepMin = 2
    }

    #endregion

    #region CONST

    public const int DFT_EPOCH_COUNT = 1;
    public const int DFT_BATCH_THREAD_COUNT = 4;
    public const int DFT_BATCH_SIZE = 1;
    public const double DFT_LEARNING_RATE = 0.1D;
    public const StopCriteria DTF_STOP_CRITERIA = StopCriteria.FullLoop;

    #endregion

    #region Fields

    private int    m_EpochCount;
    private int    m_BatchSize;
    private bool   m_UseBatchParallelization;
    private int    m_MaxBatchThreadCount;
    private double m_LearningRate;
    private double m_LossStopDelta;
    private double m_StepStopValue;
    private ILossFunction  m_LossFunction;
    private StopCriteria   m_Stop;
    private IOptimizer     m_Optimizer;
    private IRegularizator m_Regularizator;
    private ILearningRateScheduler m_LearningRateScheduler;

    private int m_EpochLength;
    private int m_InputDepth;
    private int m_InputHeight;
    private int m_InputWidth;
    private int m_OutputDepth;
    private BatchContext m_BatchContext;

    private double m_IterLossValue;
    private double m_PrevLossValue;
    private double m_LossValue;
    private double m_LossDelta;

    private double m_Step2;

    private int m_Epoch;
    private int m_Batch;
    private int m_Iteration;

    private Dictionary<Class, double[]> m_ExpectedOutputs;
    private double[][]    m_Gradient;
    private double[][][,] m_Values;
    private double[][][,] m_Errors;

    #endregion

    #region .ctor

    public BackpropAlgorithm(ClassifiedSample<double[][,]> classifiedSample, ConvNet net)
      : base(classifiedSample, net)
    {
      m_EpochCount   = DFT_EPOCH_COUNT;
      m_LearningRate = DFT_LEARNING_RATE;
      m_Stop         = DTF_STOP_CRITERIA;
      m_BatchSize    = DFT_BATCH_SIZE;
      m_MaxBatchThreadCount = DFT_BATCH_THREAD_COUNT;
    }

    #endregion

    #region Events

    public event EventHandler EpochEndedEvent;
    public event EventHandler BatchEndedEvent;

    #endregion

    #region Properties

    public override string ID   { get { return "CNN_BP"; } }
    public override string Name { get { return "Convolutional Neural Network with Backpropagation"; } }

    public int InputDepth  { get { return m_InputDepth; } }
    public int InputHeight { get { return m_InputHeight; } }
    public int InputWidth  { get { return m_InputWidth; } }
    public int OutputDepth { get { return m_OutputDepth; } }

    public double IterLossValue { get { return m_IterLossValue; } }
    public double LossValue     { get { return m_LossValue; } }
    public double LossDelta     { get { return m_LossDelta; } }
    public double Step2         { get { return m_Step2; } }


    public bool UseBatchParallelization
    {
      get { return m_UseBatchParallelization; }
      set { m_UseBatchParallelization = value; }
    }

    public int MaxBatchThreadCount
    {
      get { return m_MaxBatchThreadCount; }
      set
      {
        if (value<1)
          throw new MLException("MaxBatchThreadCount must be positive");
        m_MaxBatchThreadCount = value;
      }
    }

    public ILossFunction LossFunction
    {
      get { return m_LossFunction; }
      set { m_LossFunction = value; }
    }

    public StopCriteria Stop
    {
      get { return m_Stop; }
      set { m_Stop = value; }
    }

    public IOptimizer Optimizer
    {
      get { return m_Optimizer; }
      set { m_Optimizer = value; }
    }

    public IRegularizator Regularizator
    {
      get { return m_Regularizator; }
      set { m_Regularizator = value; }
    }

    public ILearningRateScheduler LearningRateScheduler
    {
      get { return m_LearningRateScheduler; }
      set { m_LearningRateScheduler = value; }
    }

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

    public int BatchSize
    {
      get { return m_BatchSize; }
      set
      {
        if (value <= 0)
          throw new MLException("Batch size must be positive");
        m_BatchSize = value;
      }
    }

    public double LearningRate
    {
      get { return m_LearningRate; }
      set
      {
        if (value <= 0)
          throw new MLException("Learning rate must be positive");
        m_LearningRate = value;
      }
    }

    public double LossStopDelta
    {
      get { return m_LossStopDelta; }
      set
      {
        if (LossStopDelta < 0)
          throw new MLException("Loss Stop Delta must be non-negative");
        m_LossStopDelta = value;
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

    public double[][] Gradient  { get { return m_Gradient; } }
    public double[][][,] Errors { get { return m_Errors;   } }
    public double[][][,] Values { get { return m_Values; } }

    public int Epoch     { get { return m_Epoch; } }
    public int Batch     { get { return m_Batch; } }
    public int Iteration { get { return m_Iteration; } }

    #endregion

    #region Public

    public void RunEpoch()
    {
      runEpoch();
    }

    public void RunBatch(int skip, int take)
    {
      if (skip<0) throw new MLException("Skip value must be non-negative");
      if (take<=0) throw new MLException("Take value must be positive");

      var batch = TrainingSample.Subset(skip, take);
      runBatch(batch);
    }

    public void RunIteration(double[][,] data, Class cls)
    {
      runIteration(data, cls);
    }

    public void FlushGradient()
    {
      m_Optimizer.Push(Net.Weights, m_Gradient, m_LearningRate);
    }

    public override void Build()
    {
      // init fields

      m_EpochLength = TrainingSample.Count;
      m_InputDepth  = Net.InputDepth;
      m_InputHeight = Net.InputHeight;
      m_InputWidth  = Net.InputWidth;
      m_OutputDepth = Net[Net.LayerCount - 1].OutputDepth;

      m_Gradient = new double[Net.LayerCount][];
      m_Values   = new double[Net.LayerCount][][,];
      m_Errors   = new double[Net.LayerCount][][,];
      for (int l=0; l<Net.LayerCount; l++)
      {
        var layer = Net[l];

        m_Gradient[l] = new double[layer.ParamCount];
        m_Values[l]   = new double[layer.OutputDepth][,];
        m_Errors[l]   = new double[layer.OutputDepth][,];
        for (int p=0; p<layer.OutputDepth; p++)
        {
          m_Values[l][p] = new double[layer.OutputHeight, layer.OutputWidth];
          m_Errors[l][p] = new double[layer.OutputHeight, layer.OutputWidth];
        }
      }

      // init expected outputs

      m_ExpectedOutputs = new Dictionary<Class, double[]>();
      var count = Classes.Count;
      if (count != OutputDepth)
        throw new MLException("Number of classes must be equal to dimension of output vector");

      for (int i=0; i<count; i++)
      {
        var cls = Classes.FirstOrDefault(p => (int)p.Value.Value == i).Value;
        if (cls == null)
          throw new MLException(string.Format("There is no class with value {0}. It is neccessary to have full set of classes with values from 0 to {1}", i, count));

        var output = new double[count];
        output[i] = 1.0D;
        m_ExpectedOutputs[cls] = output;
      }

      // init optimizer
      if (m_Optimizer==null)
        m_Optimizer = Registry.Optimizer.SGD;

      // init scheduler
      if (m_LearningRateScheduler==null)
        m_LearningRateScheduler = Registry.LearningRateScheduler.Constant(m_LearningRate);

      // init batch context
      if (m_UseBatchParallelization)
        m_BatchContext = new BatchContext(this, m_MaxBatchThreadCount);
    }

    #endregion

    protected override void DoTrain()
    {
      doTrain();
    }

    #region .pvt

    private void doTrain()
    {
      for (int epoch=0; epoch<m_EpochCount; epoch++)
      {
        runEpoch();
        if (checkStopCriteria()) break;
      }
    }

    private void runEpoch()
    {
      // loop on batches
      //int b = 0;
      foreach (var batch in TrainingSample.Batch(m_BatchSize))
      {
        runBatch(batch);
        //if ((++b) % 10000 == 0) Console.WriteLine("Batch: {0} ({1} iters)", b, b*BatchSize);
      }

      // update epoch stats
      m_Epoch++;
      m_Iteration = 0;
      m_Batch = 0;
      m_LearningRate = m_LearningRateScheduler.GetRate(m_Epoch);

      if (EpochEndedEvent != null) EpochEndedEvent(this, EventArgs.Empty);
    }

    private void runBatch(ClassifiedSample<double[][,]> sampleBatch)
    {
      // loop over batch
      if (m_UseBatchParallelization)
      {
        Parallel.ForEach(sampleBatch, pdata => m_BatchContext.Push(pdata.Key, pdata.Value));
      }
      else
      {
        foreach (var pdata in sampleBatch)
          runIteration(pdata.Key, pdata.Value);
      }

      // regularization
      if (m_Regularizator != null)
        m_Regularizator.Apply(m_Gradient, Net.Weights);

      // optimize and apply updates
      m_Optimizer.Push(Net.Weights, m_Gradient, m_LearningRate);

      // update batch stats
      m_Iteration += m_BatchSize;
      m_Batch++;
      m_Step2 = m_Optimizer.Step2;
      m_PrevLossValue = m_LossValue;
      m_LossValue = m_IterLossValue;
      m_LossDelta = m_LossValue - m_PrevLossValue;
      m_IterLossValue = 0.0D;

      if (BatchEndedEvent != null) BatchEndedEvent(this, EventArgs.Empty);
    }

    private void runIteration(double[][,] input, Class cls)
    {
      // feed forward
      var iterLoss = feedForward(input, cls);

      // feed backward
      var lcount = Net.LayerCount;
      for (int i=lcount-1; i>=0; i--)
      {
        var layer  = Net[i];
        var error  = m_Errors[i];
        var player = Net[i-1];
        var pvalue = (i>0) ? m_Values[i-1] : input;
        var perror = (i>0) ? m_Errors[i-1] : null;
        var gradient = m_Gradient[i];

        // error backpropagation
        if (i>0)
          layer.Backprop(player, pvalue, perror, error);

        // prepare gradient updates
        layer.SetLayerGradient(pvalue, error, gradient, true);
      }

      // update iter stats
      m_IterLossValue += iterLoss;
    }

    private double feedForward(double[][,] input, Class cls)
    {
      Net.Calculate(input, m_Values);

      var lidx   = Net.LayerCount - 1;
      var result = m_Values[lidx];
      var errors = m_Errors[lidx];
      var len    = result.GetLength(0);
      var output = new double[len];
      for (int j=0; j<len; j++) output[j] = result[j][0, 0];

      var expect = m_ExpectedOutputs[cls];
      var llayer = Net[lidx];

      for (int p=0; p<llayer.OutputDepth; p++)
      {
        var ej = m_LossFunction.Derivative(p, output, expect);
        var value = result[p][0, 0];
        var deriv = (llayer.ActivationFunction != null) ? llayer.ActivationFunction.DerivativeFromValue(value) : 1;
        errors[p][0, 0] = ej * deriv / m_BatchSize;
      }

      var loss = m_LossFunction.Value(output, expect) / m_BatchSize;
      if (m_Regularizator != null)
        loss += (m_Regularizator.Value(Net.Weights) / m_BatchSize);

      return loss;
    }

    private bool checkStopCriteria()
    {
      switch (Stop)
      {
        case StopCriteria.FullLoop:  return false;
        case StopCriteria.ErrorFunc: return Math.Abs(m_LossDelta) < m_LossStopDelta;
        case StopCriteria.StepMin:   return m_Step2 < StepStopValue;
        default: throw new MLException("Unknown stop criteria");
      }
    }

    #endregion
  }
}
