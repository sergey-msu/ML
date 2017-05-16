using System;
using System.IO;
using System.Linq;
using ML.Core;
using ML.DeepMethods.Algorithms;

namespace ML.DeepTests
{
  public abstract class Runner
  {
    public const string DFT_ROOT = @"C:\Users\User\Desktop\science\Machine learning";

    protected ClassifiedSample<double[][,]> m_TrainingSet = new ClassifiedSample<double[][,]>();
    protected ClassifiedSample<double[][,]> m_TestingSet  = new ClassifiedSample<double[][,]>();
    protected ClassifiedSample<double[][,]> m_ValidationSet; // part of a training set

    public string RootPath
    {
      get
      {
        var args = Environment.GetCommandLineArgs();
        return args.Count()>1 ? args[1] : DFT_ROOT;
      }
    }
    public abstract string SrcMark { get; }
    public abstract string DataPath  { get; }
    public string SrcPath   { get { return DataPath+@"\src\"+SrcMark; }}
    public string TestPath  { get { return DataPath+@"\test\"+SrcMark; }}
    public string TrainPath { get { return DataPath+@"\train\"+SrcMark; }}
    public abstract string OutputPath { get; }

    private BackpropAlgorithm m_Alg;
    public  BackpropAlgorithm Alg
    {
      get
      {
        if (m_Alg==null)
          m_Alg = CreateAlgorithm(m_TrainingSet);
        return m_Alg;
      }
    }


    public void Run()
    {
      Init();

      //Export();
      Load();

      var vcnt = m_TrainingSet.Count / 20;
      m_ValidationSet = m_TrainingSet.Subset(0, vcnt);

      Train();
      //Test();
    }

    protected abstract BackpropAlgorithm CreateAlgorithm(ClassifiedSample<double[][,]> sample);

    protected virtual void Init()
    {
      var paths = new []{ RootPath, DataPath, SrcPath, TestPath, TrainPath, OutputPath };
      foreach (var path in paths)
      {
        if (!Directory.Exists(path))
          Directory.CreateDirectory(path);
      }
    }

    protected abstract void Export();

    protected abstract void Load();

    protected abstract void Train();

    protected abstract void Test();
  }
}
