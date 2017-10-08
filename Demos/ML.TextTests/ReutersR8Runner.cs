using System;
using System.Collections.Generic;
using System.IO;
using ML.Core;
using ML.TextMethods.Algorithms;

namespace ML.TextTests
{
  public class ReutersR8Runner : Runner
  {
    public const string ACQ      = "acq";
    public const string CRUDE    = "crude";
    public const string EARN     = "earn";
    public const string GRAIN    = "grain";
    public const string INTEREST = "interest";
    public const string MONEY    = "money-fx";
    public const string SHIP     = "ship";
    public const string TRADE    = "trade";

    public readonly char[] SEPARATOR = new[] { '\t' };

    private Dictionary<string, Class> m_Classes = new Dictionary<string, Class>()
    {
      { ACQ,      new Class("acq",      0) },
      { CRUDE,    new Class("crude",    1) },
      { EARN,     new Class("earn",     2) },
      { GRAIN,    new Class("grain",    3) },
      { INTEREST, new Class("interest", 4) },
      { MONEY,    new Class("money-fx", 5) },
      { SHIP,     new Class("ship",     6) },
      { TRADE,    new Class("trade",    7) }
    };

    public override string SrcMark    { get { return "original"; } }
    public override string DataPath   { get { return RootPath+@"\data\reuters-r8"; } }
    public override string OutputPath { get { return RootPath+@"\output\reuters-r8_original"; } }

    protected override TextAlgorithmBase CreateAlgorithm()
    {
      return //Examples.Create_GeneralTextAlgorithm();
             Examples.Create_FourierGeneralTextAlgorithm(0.0);
             //Examples.Create_ReutersR8();
    }

    //protected override IEnumerable<TextAlgorithmBase> CreateAlgorithms()
    //{
    //  for (double t=0; t<6; t += 0.1)
    //    yield return Examples.Create_FourierTFIDFAlgorithm(t);
    //}

    #region Export

    protected override void Export()
    {
      throw new NotImplementedException();
    }

    #endregion

    #region Load

    protected override void Load()
    {
      Console.WriteLine("load train data...");
      var trainPath = Path.Combine(SrcPath, "train.txt");
      doLoad(trainPath, m_TrainingSet);

      Console.WriteLine("load test data...");
      var testPath = Path.Combine(SrcPath, "test.txt");
      doLoad(testPath, m_TestingSet);
    }

    private void doLoad(string path, ClassifiedSample<string> sample)
    {
      using (var srcFile = File.Open(path, FileMode.Open, FileAccess.Read))
      using (var srcReader = new StreamReader(srcFile))
      {
        while (true)
        {
          var line = srcReader.ReadLine();
          if (line==null) break;

          var segs = line.Split(SEPARATOR);
          var cls = m_Classes[segs[0].Trim()];
          var doc = segs[1].Trim();

          sample[doc] = cls;
        }
      }
    }

    #endregion
  }
}
