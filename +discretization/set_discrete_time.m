%
% superclass for all discrete time sets
%
% author: Martin F. Schiffner
% date: 2019-01-16
% modified: 2019-01-28
%
classdef set_discrete_time

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        T ( 1, : ) physical_values.time	% set consists of multiple times
    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = set_discrete_time( intervals_t, f_s )

            if nargin == 0
                return;
            end

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % TODO: introduce class discretized time interval
            % ensure class physical_values.time_interval
            if ~isa( intervals_t, 'physical_values.time_interval' )
                errorStruct.message     = 'intervals_t must be physical_values.time_interval!';
                errorStruct.identifier	= 'set_discrete_time:NoTimeInterval';
                error( errorStruct );
            end
            % assertion: intervals_t is physical_values.time_interval

            % ensure class physical_values.frequency
            if ~isa( f_s, 'physical_values.frequency' )
                errorStruct.message     = 'frequency must be physical_values.frequency!';
                errorStruct.identifier	= 'set_discrete_time:NoFrequency';
                error( errorStruct );
            end
            % assertion: f_s is physical_values.frequency

            % ensure identical sizes
            if ~all( size( intervals_t ) == size( f_s ) )
                errorStruct.message     = 'intervals_t and f_s must have the same size!';
                errorStruct.identifier	= 'set_discrete_time:SizeMismatch';
                error( errorStruct );
            end
            % assertion: intervals_t and f_s have the same size

            %--------------------------------------------------------------
            % 2.) compute sets of discrete time instants
            %--------------------------------------------------------------
            % create column vector of objects
            N_objects = numel( intervals_t );
            objects = repmat( objects, [ N_objects, 1 ] );

            % set independent properties
            for index_object = 1:N_objects

                % compute lower and upper bounds on the time index
                q_lb = floor( f_s( index_object ).value * intervals_t( index_object ).bounds( 1 ).value );
                q_ub = ceil( f_s( index_object ).value * intervals_t( index_object ).bounds( 2 ).value );

                % compute discrete times
                objects( index_object ).T = physical_values.time( (q_lb:(q_ub - 1)) / f_s( index_object ).value );
            end

            % reshape to sizes of the arguments
            objects = reshape( objects, size( intervals_t ) );

        end % function objects = set_discrete_time( intervals_t, f_s )

	end % methods

end % classdef set_discrete_time
