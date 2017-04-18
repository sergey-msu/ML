using System;
using System.Linq;
using ML.Core;
using ML.DeepMethods.Algorithms;

namespace ML.DeepTests
{
  public abstract class Runner
  {
    public const string DFT_ROOT = @"C:\Users\User\Desktop\science\Machine learning";

    public ClassifiedSample<double[][,]> m_Training = new ClassifiedSample<double[][,]>();

    public string Root
    {
      get
      {
        var args = Environment.GetCommandLineArgs();
        return args.Count()>1 ? args[1] : DFT_ROOT;
      }
    }

    public abstract string ResultsFolder { get; }

    public BackpropAlgorithm Alg { get; protected set; }


    public void Run()
    {
      Init();

      //Export();
      Load();
      Train();
      //Test();
    }

    protected virtual void Init()
    {
    }

    protected abstract void Export();

    protected abstract void Load();

    protected abstract void Train();

    protected abstract void Test();
  }
}
