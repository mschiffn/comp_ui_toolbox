%
% superclass for all strictly monotonically increasing sequences w/
% regular spacing
%
% author: Martin F. Schiffner
% date: 2020-01-11
% modified: 2020-01-11
%
classdef sequence_increasing_regular < math.sequence_increasing

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        offset ( 1, 1 ) physical_values.physical_quantity { mustBeNonempty } = physical_values.second( 0 )                  % arbitrary offset
        delta ( 1, 1 ) physical_values.physical_quantity { mustBePositive, mustBeNonempty } = physical_values.second( 1 )	% step size

    end % properties

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = sequence_increasing_regular( offsets, deltas, N_members )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure equal subclasses of physical_values.physical_quantity
            auxiliary.mustBeEqualSubclasses( 'physical_values.physical_quantity', offsets, deltas );

            % property validation function ensures positive integers for N_members

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( offsets, deltas, N_members );

            %--------------------------------------------------------------
            % 2.) create strictly monotonically increasing sequences (regular spacing)
            %--------------------------------------------------------------
            % specify cell array for members
            members = cell( size( offsets ) );

            % iterate sequences
            for index_object = 1:numel( offsets )

                % compute members
                members{ index_object } = offsets( index_object ) + ( 0:( N_members( index_object ) - 1 ) )' * deltas( index_object );

            end % for index_object = 1:numel( offsets )

            % constructor of superclass
            objects@math.sequence_increasing( members );

            % iterate sequences
            for index_object = 1:numel( objects )

                % set independent properties
                objects( index_object ).offset = offsets( index_object );
                objects( index_object ).delta = deltas( index_object );

            end % for index_object = 1:numel( objects )

        end % function objects = sequence_increasing_regular( offsets, deltas, N_members )

    end % methods

end % classdef sequence_increasing_regular < math.sequence_increasing
