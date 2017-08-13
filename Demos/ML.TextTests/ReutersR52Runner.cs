using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using ML.Core;
using ML.TextMethods.Algorithms;

namespace ML.TextTests
{
  public class ReutersR52Runner : Runner
  {
    public const string ACQ             = "acq";
    public const string ALUM            = "alum";
    public const string BOP             = "bop";
    public const string CARCASS         = "carcass";
    public const string COCOA           = "cocoa";
    public const string COFFEE          = "coffee";
    public const string COPPER          = "copper";
    public const string COTTON          = "cotton";
    public const string CPI             = "cpi";
    public const string CPU             = "cpu";
    public const string CRUDE           = "crude";
    public const string DLR             = "dlr";
    public const string EARN            = "earn";
    public const string FUEL            = "fuel";
    public const string GAS             = "gas";
    public const string GNP             = "gnp";
    public const string GOLD            = "gold";
    public const string GRAIN           = "grain";
    public const string HEAT            = "heat";
    public const string HOUSING         = "housing";
    public const string INCOME          = "income";
    public const string INSTAL_DEBT     = "instal-debt";
    public const string INTEREST        = "interest";
    public const string IPI             = "ipi";
    public const string IRON_STEEL      = "iron-steel";
    public const string JET             = "jet";
    public const string JOBS            = "jobs";
    public const string LEAD            = "lead";
    public const string LEI             = "lei";
    public const string LIVESTOCK       = "livestock";
    public const string LUMBER          = "lumber";
    public const string MEAL_FEED       = "meal-feed";
    public const string MONEY_FX        = "money-fx";
    public const string MONEY_SUPPLY    = "money-supply";
    public const string NAT_GAS         = "nat-gas";
    public const string NICKEL          = "nickel";
    public const string ORANGE          = "orange";
    public const string PET_CHEM        = "pet-chem";
    public const string PLATINUM        = "platinum";
    public const string POTATO          = "potato";
    public const string RESERVES        = "reserves";
    public const string RETAIL          = "retail";
    public const string RUBBER          = "rubber";
    public const string SHIP            = "ship";
    public const string STRATEGIC_METAL = "strategic-metal";
    public const string SUGAR           = "sugar";
    public const string TEA             = "tea";
    public const string TIN             = "tin";
    public const string TRADE           = "trade";
    public const string VEG_OIL         = "veg-oil";
    public const string WPI             = "wpi";
    public const string ZINC            = "zinc";

    public readonly char[] SEPARATOR = new[] { '\t' };

    private Dictionary<string, Class> m_Classes = new Dictionary<string, Class>()
    {
      { ACQ,             new Class("acq",             0)  },
      { ALUM,            new Class("alum",            1)  },
      { BOP,             new Class("bop",             2)  },
      { CARCASS,         new Class("carcass",         3)  },
      { COCOA,           new Class("cocoa",           4)  },
      { COFFEE,          new Class("coffee",          5)  },
      { COPPER,          new Class("copper",          6)  },
      { COTTON,          new Class("cotton",          7)  },
      { CPI,             new Class("cpi",             8)  },
      { CPU,             new Class("cpu",             9)  },
      { CRUDE,           new Class("crude",           10) },
      { DLR,             new Class("dlr",             11) },
      { EARN,            new Class("earn",            12) },
      { FUEL,            new Class("fuel",            13) },
      { GAS,             new Class("gas",             14) },
      { GNP,             new Class("gnp",             15) },
      { GOLD,            new Class("gold",            16) },
      { GRAIN,           new Class("grain",           17) },
      { HEAT,            new Class("heat",            18) },
      { HOUSING,         new Class("housing",         19) },
      { INCOME,          new Class("income",          20) },
      { INSTAL_DEBT,     new Class("instal-debt",     21) },
      { INTEREST,        new Class("interest",        22) },
      { IPI,             new Class("ipi",             23) },
      { IRON_STEEL,      new Class("iron-steel",      24) },
      { JET,             new Class("jet",             25) },
      { JOBS,            new Class("jobs",            26) },
      { LEAD,            new Class("lead",            27) },
      { LEI,             new Class("lei",             28) },
      { LIVESTOCK,       new Class("livestock",       29) },
      { LUMBER,          new Class("lumber",          30) },
      { MEAL_FEED,       new Class("meal-feed",       31) },
      { MONEY_FX,        new Class("money-fx",        32) },
      { MONEY_SUPPLY,    new Class("money-supply",    33) },
      { NAT_GAS,         new Class("nat-gas",         34) },
      { NICKEL,          new Class("nickel",          35) },
      { ORANGE,          new Class("orange",          36) },
      { PET_CHEM,        new Class("pet-chem",        37) },
      { PLATINUM,        new Class("platinum",        38) },
      { POTATO,          new Class("potato",          39) },
      { RESERVES,        new Class("reserves",        40) },
      { RETAIL,          new Class("retail",          41) },
      { RUBBER,          new Class("rubber",          42) },
      { SHIP,            new Class("ship",            43) },
      { STRATEGIC_METAL, new Class("strategic-metal", 44) },
      { SUGAR,           new Class("sugar",           45) },
      { TEA,             new Class("tea",             46) },
      { TIN,             new Class("tin",             47) },
      { TRADE,           new Class("trade",           48) },
      { VEG_OIL,         new Class("veg-oil",         49) },
      { WPI,             new Class("wpi",             50) },
      { ZINC,            new Class("zinc",            51) },
    };

    public override string SrcMark    { get { return "original"; } }
    public override string DataPath   { get { return RootPath+@"\data\reuters-r52"; }}
    public override string OutputPath { get { return RootPath+@"\output\reuters-r52_original"; }}

    protected override TextAlgorithmBase CreateAlgorithm()
    {
      return Examples.Create_TWCAlgorithm();
    }

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
