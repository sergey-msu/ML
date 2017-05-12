using System;
using System.Linq;
using ML.Core;
using ML.DeepMethods.Algorithms;
using System.IO;

namespace ML.DeepTests
{
  public abstract class Runner
  {
    public const string DFT_ROOT = @"C:\Users\User\Desktop\science\Machine learning";

    protected ClassifiedSample<double[][,]> m_TrainingSet = new ClassifiedSample<double[][,]>();
    protected ClassifiedSample<double[][,]> m_TestingSet  = new ClassifiedSample<double[][,]>();
    protected ClassifiedSample<double[][,]> m_ValidationSet;

    public string RootPath
    {
      get
      {
        var args = Environment.GetCommandLineArgs();
        return args.Count()>1 ? args[1] : DFT_ROOT;
      }
    }
    public abstract string DataPath  { get; }
    public string SrcPath   { get { return DataPath+@"\src\original"; }}
    public string TestPath  { get { return DataPath+@"\test\original"; }}
    public string TrainPath { get { return DataPath+@"\train\original"; }}
    public abstract string OutputPath { get; }

    public BackpropAlgorithm Alg { get; protected set; }


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
