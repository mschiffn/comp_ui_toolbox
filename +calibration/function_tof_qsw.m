function [ tof, tof_sc ] = function_tof_qsw( positions_lateral, states )
%
% compute round-trip times-of-flight for a quasi-spherical wave
%
% positions_lateral: lateral positions of array elements
% states: state objects containing target positions and average sound speed
% tof: times-of-flight
%
% author: Martin F. Schiffner
% date: 2019-06-12
% modified: 2020-01-21

    %----------------------------------------------------------------------
    % 1.) check arguments
    %----------------------------------------------------------------------
    % ensure cell array for positions_lateral
	if ~iscell( positions_lateral )
        positions_lateral = { positions_lateral };
    end

	% ensure class calibration.state
    if ~isa( states, 'calibration.state' )
        errorStruct.message = 'states must be calibration.state!';
        errorStruct.identifier = 'function_tof_qsw:NoStates';
        error( errorStruct );
    end

	% multiple positions_lateral / single states
	if ~isscalar( positions_lateral ) && isscalar( states )
        states = repmat( states, size( positions_lateral ) );
    end

	% single positions_lateral / multiple states
	if isscalar( positions_lateral ) && ~isscalar( states )
        positions_lateral = repmat( positions_lateral, size( states ) );
    end

	% ensure equal number of dimensions and sizes
	auxiliary.mustBeEqualSize( positions_lateral, states );

	%----------------------------------------------------------------------
	% 2.) compute times-of-flight
	%----------------------------------------------------------------------
	% specify cell array for tof
	tof = cell( size( positions_lateral ) );
    tof_sc = cell( size( positions_lateral ) );

	% iterate states
	for index_state = 1:numel( positions_lateral )

        % compute distances
        dist = vecnorm( [ positions_lateral{ index_state }, zeros( size( positions_lateral{ index_state }, 1 ), 1 ) ] - states( index_state ).position_target, 2, 2 );

        % compute scattered times-of-flight
        tof_sc{ index_state } = dist / states( index_state ).c_avg;

        % compute round-trip times-of-flight
        tof{ index_state } = tof_sc{ index_state } + tof_sc{ index_state }';

    end % for index_state = 1:numel( positions_lateral )

    % avoid cell array for single positions_lateral
	if isscalar( positions_lateral )
        tof = tof{ 1 };
        tof_sc = tof_sc{ 1 };
    end

end % function tof = function_tof_qsw( positions_lateral, states )
