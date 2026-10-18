#if canImport(Testing)
import Testing
import CandleDatasets

@Suite("CandleDatasets Swift Export Tests")
struct CandleDatasetsExportTests {
    @Test("CandleDatasets swift module imported cleanly")
    func testSwiftModuleLoads() throws {
        #expect(Bool(true), "CandleDatasets swift module imported cleanly")
    }
}
#elseif canImport(XCTest)
import XCTest
import CandleDatasets

final class CandleDatasetsExportTests: XCTestCase {
    func testSwiftModuleLoads() throws {
        XCTAssertTrue(true, "CandleDatasets swift module imported cleanly")
    }
}
#endif
