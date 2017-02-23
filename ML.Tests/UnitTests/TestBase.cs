using System;
using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;


namespace ML.Tests.UnitTests
{
  [TestClass]
  public class TestBase
  {
    private static TestContext m_Context;

    public static TestContext Context { get { return m_Context; } }

    [ClassInitialize]
    public static void BaseClassInit(TestContext context)
    {
      m_Context = context;
    }
  }
}
