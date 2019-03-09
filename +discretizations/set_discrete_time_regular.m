%
% superclass for all regular discrete time sets
%
% author: Martin F. Schiffner
% date: 2019-02-21
% modified: 2019-02-21
%
classdef set_discrete_time_regular < discretizations.set_discrete_time

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        q_lb ( 1, 1 ) int64                 % lower integer bound
        q_ub ( 1, 1 ) int64                 % upper integer bound
        T_s ( 1, 1 ) physical_values.time	% sampling period

	end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = set_discrete_time_regular( lbs_q, ubs_q, T_s )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure integer bounds
            if ~( all( lbs_q( : ) == floor( lbs_q( : ) ) ) && all( ubs_q( : ) == floor( ubs_q( : ) ) ) )
                errorStruct.message     = 'Boundary indices must be integers!';
                errorStruct.identifier	= 'set_discrete_time_regular:NoIntegerBounds';
                error( errorStruct );
            end

            % ensure class physical_values.time
            if ~isa( T_s, 'physical_values.time' )
                errorStruct.message     = 'T_s must be physical_values.time!';
                errorStruct.identifier	= 'set_discrete_time_regular:NoTime';
                error( errorStruct );
            end

            % multiple integer bounds, single sampling period
            if ~isscalar( lbs_q ) && isscalar( T_s )
                T_s = repmat( T_s, size( lbs_q ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( lbs_q, ubs_q, T_s );
            % assertion: lbs_q, ubs_q, and T_s have equal sizes

            %--------------------------------------------------------------
            % 2.) construct regular sets of discrete times
            %--------------------------------------------------------------
            N_objects = numel( lbs_q );
            sets = cell( size( lbs_q ) );
            for index_object = 1:N_objects
                sets{ index_object } = ( lbs_q( index_object ):ubs_q( index_object ) ) .* T_s( index_object );
            end

            %--------------------------------------------------------------
            % 3.) constructor of superclass
            %--------------------------------------------------------------
            objects@discretizations.set_discrete_time( sets );

            %--------------------------------------------------------------
            % 4.) set independent properties
            %--------------------------------------------------------------
            for index_object = 1:N_objects
                objects( index_object ).T_s = T_s( index_object );
                objects( index_object ).q_lb = int64( lbs_q( index_object ) );
                objects( index_object ).q_ub = int64( ubs_q( index_object ) );
            end

        end % function objects = set_discrete_time_regular( lbs_q, ubs_q, T_s )

	end % methods

end % classdef set_discrete_time_regular
