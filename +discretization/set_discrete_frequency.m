%
% superclass for all discrete frequency sets
%
% author: Martin F. Schiffner
% date: 2019-01-16
% modified: 2019-01-28
%
classdef set_discrete_frequency

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        F_BP ( 1, : ) physical_values.frequency	% set consists of multiple frequencies
    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = set_discrete_frequency( intervals_f, T_rec )

            if nargin == 0
                return;
            end

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % TODO: introduce class discretized frequency interval
            % ensure class physical_values.frequency_interval
            if ~isa( intervals_f, 'physical_values.frequency_interval' )
                errorStruct.message     = 'intervals_f must be physical_values.frequency_interval!';
                errorStruct.identifier	= 'set_discrete_frequency:NoTimeInterval';
                error( errorStruct );
            end
            % assertion: intervals_f is physical_values.frequency_interval

            % ensure class physical_values.time
            if ~isa( T_rec, 'physical_values.time' )
                errorStruct.message     = 'frequency must be physical_values.time!';
                errorStruct.identifier	= 'set_discrete_frequency:NoFrequency';
                error( errorStruct );
            end
            % assertion: T_rec is physical_values.time

            % ensure identical sizes
            if ~all( size( intervals_f ) == size( T_rec ) )
                errorStruct.message     = 'intervals_f and T_rec must have the same size!';
                errorStruct.identifier	= 'set_discrete_frequency:SizeMismatch';
                error( errorStruct );
            end
            % assertion: intervals_f and T_rec have the same size

            %--------------------------------------------------------------
            % 2.) compute sets of discrete frequencies
            %--------------------------------------------------------------
            % create column vector of objects
            N_objects = numel( intervals_f );
            objects = repmat( objects, [ N_objects, 1 ] );

            % set independent properties
            for index_object = 1:N_objects

                % compute lower and upper bounds on the frequency index
                l_lb = ceil( T_rec( index_object ) * double( intervals_f( index_object ).bounds( 1 ) ) );
                l_ub = floor( T_rec( index_object ) * double( intervals_f( index_object ).bounds( 2 ) ) );

                % compute discrete frequencies
                objects( index_object ).F_BP = physical_values.frequency( (l_lb:l_ub) / T_rec( index_object ) );
            end

            % reshape to sizes of the arguments
            objects = reshape( objects, size( intervals_f ) );

        end % function objects = set_discrete_frequency( intervals_f, T_rec )

	end % methods

end % classdef set_discrete_frequency
