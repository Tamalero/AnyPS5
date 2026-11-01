#include "SpirvBackend/SpirvBda.hpp"
#include "SpirvBackend/SpirvMemory/SpirvTypes.hpp"
#include "SpirvBackend/SpirvMemory/SpirvConstants.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <stdexcept>
#include <string_view>

namespace ShaderRecompiler {

std::uint32_t BdaConstant(SpirvEmitterState& state, std::uint64_t value) {
    return state.module.Constant(spv::OpConstant, TypeScalarU64(state), static_cast<std::uint32_t>(value), static_cast<std::uint32_t>(value >> 32u));
}

std::uint32_t BdaWord(SpirvEmitterState& state, std::uint32_t variable, std::uint32_t index) {
    const auto pointer = state.module.AllocateId();
    state.module.AddFunction(spv::OpAccessChain, TypeStorageBufferElementPointer(state), pointer, variable, ConstantU32(state, 0u), index);
    return pointer;
}

std::uint32_t BdaLoadWord(SpirvEmitterState& state, std::uint32_t index) {
    const auto value = state.module.AllocateId();
    state.module.AddFunction(spv::OpLoad, TypeU32(state), value, BdaWord(state, state.bdaPagetableVariable, index));
    return value;
}

std::uint32_t BdaLoadAddress(SpirvEmitterState& state, std::uint32_t index) {
    const auto low = Unary(state, spv::OpUConvert, TypeScalarU64(state), BdaLoadWord(state, index));
    const auto high = Unary(state, spv::OpUConvert, TypeScalarU64(state), BdaLoadWord(state, Binary(state, spv::OpIAdd, TypeU32(state), index, ConstantU32(state, 1u))));
    return Binary(state, spv::OpBitwiseOr, TypeScalarU64(state), low, Binary(state, spv::OpShiftLeftLogical, TypeScalarU64(state), high, BdaConstant(state, 32u)));
}

// APS5_SPIRV_FAULT_DEFER: 1 = keep the first fault; cost tests only: 2 = record nothing, 3 = keep only the reason.
int DeferMode() {
    static const int mode = [] {
        const char* value = std::getenv("APS5_SPIRV_FAULT_DEFER");
        return value == nullptr ? 0 : std::atoi(value);
    }();
    return mode;
}

void DefineBdaFaultFunction(SpirvEmitterState& state) {
    const auto u32 = TypeU32(state);
    const bool words = !state.program.Info().usesDma;
    const auto addressType = words ? u32 : TypeScalarU64(state);
    state.bdaFaultFunction = state.module.AllocateId();
    state.module.AddName(state.bdaFaultFunction, "record_bda_fault");
    const auto type = words ? state.module.Type(spv::OpTypeFunction, TypeVoid(state), u32, addressType, u32, u32, u32) : state.module.Type(spv::OpTypeFunction, TypeVoid(state), u32, addressType, u32, u32);
    state.module.AddFunction(spv::OpFunction, TypeVoid(state), state.bdaFaultFunction, spv::FunctionControlDontInlineMask, type);
    const auto reason = state.module.AllocateId();
    const auto address = state.module.AllocateId();
    const auto addressHigh = words ? state.module.AllocateId() : 0u;
    const auto bytes = state.module.AllocateId();
    const auto instruction = state.module.AllocateId();
    state.module.AddFunction(spv::OpFunctionParameter, u32, reason);
    state.module.AddFunction(spv::OpFunctionParameter, addressType, address);
    if (words) state.module.AddFunction(spv::OpFunctionParameter, u32, addressHigh);
    state.module.AddFunction(spv::OpFunctionParameter, u32, bytes);
    state.module.AddFunction(spv::OpFunctionParameter, u32, instruction);
    EmitLabel(state, state.module.AllocateId());
    const auto pointer = BdaWord(state, state.faultBufferVariable, ConstantU32(state, 0u));
    const auto previous = state.module.AllocateId();
    const auto scope = ConstantU32(state, spv::ScopeDevice);
    const auto relaxed = ConstantU32(state, spv::MemorySemanticsMaskNone);
    const auto notUnaligned = Binary(state, spv::OpINotEqual, TypeBool(state), reason, ConstantU32(state, static_cast<std::uint32_t>(BdaAbi::FaultReason::Unaligned)));
    const auto otherFaults = ConstantU32(state, static_cast<std::uint32_t>(offsetof(BdaAbi::Fault, otherFaults) / sizeof(std::uint32_t)));
    // APS5_SPIRV_FAULT_FLAT=1 (prototype for #13538): OR 0 or 1 unconditionally instead of branching; this function is
    // inlined tens of thousands of times into large shaders, and each branch is a nested if for RADV.
    static const bool flat = [] {
        const char* value = std::getenv("APS5_SPIRV_FAULT_FLAT");
        return value != nullptr && std::string_view(value) == "1";
    }();
    if (flat) {
        const auto bit = state.module.AllocateId();
        state.module.AddFunction(spv::OpSelect, u32, bit, notUnaligned, ConstantU32(state, 1u), ConstantU32(state, 0u));
        state.module.AddFunction(spv::OpAtomicOr, u32, state.module.AllocateId(), BdaWord(state, state.faultBufferVariable, otherFaults), scope, relaxed, bit);
    } else {
        EmitIfCondition(state, notUnaligned, [&] {
            state.module.AddFunction(spv::OpAtomicOr, u32, state.module.AllocateId(), BdaWord(state, state.faultBufferVariable, otherFaults), scope, relaxed, ConstantU32(state, 1u));
        });
    }
    state.module.AddFunction(spv::OpAtomicCompareExchange, TypeU32(state), previous, pointer, scope, relaxed, relaxed, ConstantU32(state, static_cast<std::uint32_t>(BdaAbi::FaultState::Writing)), ConstantU32(state, 0u));
    const auto won = Binary(state, spv::OpIEqual, TypeBool(state), previous, ConstantU32(state, 0u));
    EmitIfCondition(state, won, [&] {
        const auto store = [&](std::uint32_t index, std::uint32_t value) {
            state.module.AddFunction(spv::OpStore, BdaWord(state, state.faultBufferVariable, ConstantU32(state, index)), value);
        };
        store(1, reason);
        store(2, words ? address : Unary(state, spv::OpUConvert, TypeU32(state), address));
        store(3, words ? addressHigh : Unary(state, spv::OpUConvert, TypeU32(state), Binary(state, spv::OpShiftRightLogical, TypeScalarU64(state), address, BdaConstant(state, 32u))));
        store(4, bytes);
        store(5, ConstantU32(state, static_cast<std::uint32_t>(state.program.Resources().stage)));
        store(6, instruction);
        const auto release = ConstantU32(state, spv::MemorySemanticsReleaseMask | spv::MemorySemanticsUniformMemoryMask);
        state.module.AddFunction(spv::OpAtomicStore, pointer, scope, release, ConstantU32(state, static_cast<std::uint32_t>(BdaAbi::FaultState::Ready)));
    });
    state.module.AddFunction(spv::OpReturn);
    state.module.AddFunction(spv::OpFunctionEnd);
    // APS5_SPIRV_FAULT_DEFER=1: the lookups defined after this record into the pending fault instead of calling here.
    const bool defer = DeferMode() != 0;
    if (defer && state.program.Resources().stage != IrShaderStage::Pixel) {
        const auto pointerType = TypePointer(state, spv::StorageClassPrivate, u32);
        state.pendingFaultReason = state.module.DefineGlobalVariable(pointerType, spv::StorageClassPrivate);
        state.pendingFaultLow = state.module.DefineGlobalVariable(pointerType, spv::StorageClassPrivate);
        state.pendingFaultHigh = state.module.DefineGlobalVariable(pointerType, spv::StorageClassPrivate);
        state.pendingFaultBytes = state.module.DefineGlobalVariable(pointerType, spv::StorageClassPrivate);
        state.pendingFaultInstruction = state.module.DefineGlobalVariable(pointerType, spv::StorageClassPrivate);
        state.pendingFaultOther = state.module.DefineGlobalVariable(pointerType, spv::StorageClassPrivate);
    }
}

namespace {

// Keeps the first fault of the invocation (a branch-free update) and notes any fault that is not Unaligned.
void DeferBdaFault(SpirvEmitterState& state, std::uint32_t reason, std::uint32_t low, std::uint32_t high, std::uint32_t bytes, std::uint32_t instruction) {
    const auto u32 = TypeU32(state);
    const auto load = [&](std::uint32_t variable) {
        const auto value = state.module.AllocateId();
        state.module.AddFunction(spv::OpLoad, u32, value, variable);
        return value;
    };
    if (DeferMode() == 2) return;
    const auto current = load(state.pendingFaultReason);
    const auto first = Binary(state, spv::OpIEqual, TypeBool(state), current, ConstantU32(state, 0u));
    const auto keep = [&](std::uint32_t variable, std::uint32_t value, std::uint32_t old) {
        const auto chosen = state.module.AllocateId();
        state.module.AddFunction(spv::OpSelect, u32, chosen, first, value, old);
        state.module.AddFunction(spv::OpStore, variable, chosen);
    };
    keep(state.pendingFaultReason, reason, current);
    if (DeferMode() == 3) return;
    keep(state.pendingFaultLow, low, load(state.pendingFaultLow));
    keep(state.pendingFaultHigh, high, load(state.pendingFaultHigh));
    keep(state.pendingFaultBytes, bytes, load(state.pendingFaultBytes));
    keep(state.pendingFaultInstruction, instruction, load(state.pendingFaultInstruction));
    const auto notUnaligned = Binary(state, spv::OpINotEqual, TypeBool(state), reason, ConstantU32(state, static_cast<std::uint32_t>(BdaAbi::FaultReason::Unaligned)));
    const auto bit = state.module.AllocateId();
    state.module.AddFunction(spv::OpSelect, u32, bit, notUnaligned, ConstantU32(state, 1u), ConstantU32(state, 0u));
    state.module.AddFunction(spv::OpStore, state.pendingFaultOther, Binary(state, spv::OpBitwiseOr, u32, load(state.pendingFaultOther), bit));
}

void DeferBdaFault64(SpirvEmitterState& state, std::uint32_t reason, std::uint32_t address, std::uint32_t bytes, std::uint32_t instruction) {
    const auto u32 = TypeU32(state);
    const auto low = Unary(state, spv::OpUConvert, u32, address);
    const auto high = Unary(state, spv::OpUConvert, u32, Binary(state, spv::OpShiftRightLogical, TypeScalarU64(state), address, BdaConstant(state, 32u)));
    DeferBdaFault(state, reason, low, high, bytes, instruction);
}

}

void ResetPendingBdaFault(SpirvEmitterState& state) {
    if (state.pendingFaultReason == 0) return;
    for (const auto variable : {state.pendingFaultReason, state.pendingFaultLow, state.pendingFaultHigh, state.pendingFaultBytes, state.pendingFaultInstruction, state.pendingFaultOther}) {
        state.module.AddFunction(spv::OpStore, variable, ConstantU32(state, 0u));
    }
}

void FlushPendingBdaFault(SpirvEmitterState& state) {
    if (state.pendingFaultReason == 0) return;
    const auto u32 = TypeU32(state);
    const auto load = [&](std::uint32_t variable) {
        const auto value = state.module.AllocateId();
        state.module.AddFunction(spv::OpLoad, u32, value, variable);
        return value;
    };
    const auto reason = load(state.pendingFaultReason);
    EmitIfCondition(state, Binary(state, spv::OpINotEqual, TypeBool(state), reason, ConstantU32(state, 0u)), [&] {
        const auto low = load(state.pendingFaultLow);
        const auto high = load(state.pendingFaultHigh);
        const auto bytes = load(state.pendingFaultBytes);
        const auto instruction = load(state.pendingFaultInstruction);
        if (state.program.Info().usesDma) {
            const auto u64 = TypeScalarU64(state);
            const auto address = Binary(state, spv::OpBitwiseOr, u64, Unary(state, spv::OpUConvert, u64, low), Binary(state, spv::OpShiftLeftLogical, u64, Unary(state, spv::OpUConvert, u64, high), BdaConstant(state, 32u)));
            state.module.AddFunction(spv::OpFunctionCall, TypeVoid(state), state.module.AllocateId(), state.bdaFaultFunction, reason, address, bytes, instruction);
        } else {
            state.module.AddFunction(spv::OpFunctionCall, TypeVoid(state), state.module.AllocateId(), state.bdaFaultFunction, reason, low, high, bytes, instruction);
        }
    });
    // record_bda_fault marks otherFaults for the first fault; a later fault that was not Unaligned marks it here.
    EmitIfCondition(state, Binary(state, spv::OpINotEqual, TypeBool(state), load(state.pendingFaultOther), ConstantU32(state, 0u)), [&] {
        state.module.AddFunction(spv::OpAtomicOr, u32, state.module.AllocateId(), BdaWord(state, state.faultBufferVariable, ConstantU32(state, static_cast<std::uint32_t>(offsetof(BdaAbi::Fault, otherFaults) / sizeof(std::uint32_t)))), ConstantU32(state, spv::ScopeDevice), ConstantU32(state, spv::MemorySemanticsMaskNone), ConstantU32(state, 1u));
    });
}

void RecordBdaFault(SpirvEmitterState& state, std::uint32_t address, std::uint32_t bytes, std::uint32_t instruction, BdaAbi::FaultReason reason) {
    if (state.bdaFaultFunction == 0 || !state.program.Info().usesDma) throw std::runtime_error("BDA fault function is missing");
    if (state.pendingFaultReason != 0) return DeferBdaFault64(state, ConstantU32(state, static_cast<std::uint32_t>(reason)), address, bytes, instruction);
    state.module.AddFunction(spv::OpFunctionCall, TypeVoid(state), state.module.AllocateId(), state.bdaFaultFunction, ConstantU32(state, static_cast<std::uint32_t>(reason)), address, bytes, instruction);
}

void RecordBdaFaultReason(SpirvEmitterState& state, std::uint32_t address, std::uint32_t bytes, std::uint32_t instruction, std::uint32_t reason) {
    if (state.bdaFaultFunction == 0 || !state.program.Info().usesDma) throw std::runtime_error("BDA fault function is missing");
    if (state.pendingFaultReason != 0) return DeferBdaFault64(state, reason, address, bytes, instruction);
    state.module.AddFunction(spv::OpFunctionCall, TypeVoid(state), state.module.AllocateId(), state.bdaFaultFunction, reason, address, bytes, instruction);
}

void RecordBdaFaultWords(SpirvEmitterState& state, std::uint32_t addressLow, std::uint32_t addressHigh, std::uint32_t bytes, std::uint32_t instruction, BdaAbi::FaultReason reason) {
    if (state.pendingFaultReason != 0) return DeferBdaFault(state, ConstantU32(state, static_cast<std::uint32_t>(reason)), addressLow, addressHigh, bytes, instruction);
    if (state.program.Info().usesDma) {
        const auto u64 = TypeScalarU64(state);
        const auto high = Binary(state, spv::OpShiftLeftLogical, u64, Unary(state, spv::OpUConvert, u64, addressHigh), BdaConstant(state, 32u));
        RecordBdaFault(state, Binary(state, spv::OpBitwiseOr, u64, Unary(state, spv::OpUConvert, u64, addressLow), high), bytes, instruction, reason);
        return;
    }
    if (state.bdaFaultFunction == 0) throw std::runtime_error("BDA fault function is missing");
    state.module.AddFunction(spv::OpFunctionCall, TypeVoid(state), state.module.AllocateId(), state.bdaFaultFunction, ConstantU32(state, static_cast<std::uint32_t>(reason)), addressLow, addressHigh, bytes, instruction);
}

void ReturnBdaFailureIf(SpirvEmitterState& state, std::uint32_t condition, std::uint32_t address, std::uint32_t bytes, std::uint32_t instruction, BdaAbi::FaultReason reason) {
    const auto failed = state.module.AllocateId();
    const auto next = state.module.AllocateId();
    state.module.AddFunction(spv::OpSelectionMerge, next, spv::SelectionControlMaskNone);
    state.module.AddFunction(spv::OpBranchConditional, condition, failed, next);
    EmitLabel(state, failed);
    RecordBdaFault(state, address, bytes, instruction, reason);
    state.module.AddFunction(spv::OpReturnValue, BdaConstant(state, 0u));
    EmitLabel(state, next);
}

void StopBdaInvocationIf(SpirvEmitterState& state, std::uint32_t condition) {
    if (!state.bdaStopsInvocations) return;
    const auto failed = state.module.AllocateId();
    const auto next = state.module.AllocateId();
    state.module.AddFunction(spv::OpSelectionMerge, next, spv::SelectionControlMaskNone);
    state.module.AddFunction(spv::OpBranchConditional, condition, failed, next);
    EmitLabel(state, failed);
    if (state.bdaStopValue != 0) {
        state.module.AddFunction(spv::OpReturnValue, state.bdaStopValue);
    } else if (state.stopTarget != 0 && state.module.InnermostOpenLoopMerge() == state.stopTarget) {
        // A stop inside a loop the emitter built within one instruction keeps its OpReturn: a branch to stopTarget
        // would leave that loop without a structured exit.
        if (state.stopTarget != state.stopExit) state.module.AddFunction(spv::OpStore, state.stopFlag, ConstantBool(state, true));
        state.stopEdges[state.stopTarget].push_back(state.currentLabel);
        state.module.AddFunction(spv::OpBranch, state.stopTarget);
    } else {
        if (OrderedPixelShader(state)) {
            state.module.AddFunction(spv::OpEndInvocationInterlockEXT);
        }
        if (state.program.Resources().stage != IrShaderStage::Pixel) {
            FlushPendingBdaFault(state);
            EmitReturnReleasingScratch(state);
        } else if (state.pixelStopFlag != 0u) {
            state.module.AddFunction(spv::OpStore, state.pixelStopFlag, ConstantBool(state, true));
            state.module.AddFunction(spv::OpReturn);
            state.pixelStopStored = true;
        } else if (std::ranges::find(state.supportedExtensions, "SPV_KHR_terminate_invocation") != state.supportedExtensions.end()) {
            state.module.EmitExtension("SPV_KHR_terminate_invocation");
            state.module.AddFunction(spv::OpTerminateInvocation);
        } else {
            state.module.AddFunction(spv::OpKill);
        }
    }
    EmitLabel(state, next);
}

}
