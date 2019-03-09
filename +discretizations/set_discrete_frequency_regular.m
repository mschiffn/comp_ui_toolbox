%
% superclass for all regular discrete frequency sets
%
% author: Martin F. Schiffner
% date: 2019-02-21
% modified: 2019-02-21
%
classdef set_discrete_frequency_regular < discretizations.set_discrete_frequency

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        q_lb ( 1, 1 ) int64                     % lower integer bound
        q_ub ( 1, 1 ) int64                     % upper integer bound
        F_s ( 1, 1 ) physical_values.frequency	% sampling period

	end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = set_discrete_frequency_regular( lbs_q, ubs_q, F_s )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure integer bounds
            if ~( all( lbs_q( : ) == floor( lbs_q( : ) ) ) && all( ubs_q( : ) == floor( ubs_q( : ) ) ) )
                errorStruct.message     = 'Boundary indices must be integers!';
                errorStruct.identifier	= 'set_discrete_frequency_regular:NoIntegerBounds';
                error( errorStruct );
            end

            % ensure class physical_values.frequency
            if ~isa( F_s, 'physical_values.frequency' )
                errorStruct.message     = 'F_s must be physical_values.frequency!';
                errorStruct.identifier	= 'set_discrete_frequency_regular:NoFrequency';
                error( errorStruct );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( lbs_q, ubs_q, F_s );
            % assertion: lbs_q, ubs_q, and F_s have equal sizes

            %--------------------------------------------------------------
            % 2.) construct regular sets of discrete frequencies
            %--------------------------------------------------------------
            N_objects = numel( lbs_q );
            sets = cell( size( lbs_q ) );
            for index_object = 1:N_objects
                sets{ index_object } = ( lbs_q( index_object ):ubs_q( index_object ) ) .* F_s( index_object );
            end

            %--------------------------------------------------------------
            % 3.) constructor of superclass
            %--------------------------------------------------------------
            objects@discretizations.set_discrete_frequency( sets );

            %--------------------------------------------------------------
            % 4.) set independent properties
            %--------------------------------------------------------------
            for index_object = 1:N_objects
                objects( index_object ).F_s = F_s( index_object );
                objects( index_object ).q_lb = int64( lbs_q( index_object ) );
                objects( index_object ).q_ub = int64( ubs_q( index_object ) );
            end

        end % function objects = set_discrete_frequency_regular( lbs_q, ubs_q, F_s )

	end % methods

end % classdef set_discrete_frequency_regular
