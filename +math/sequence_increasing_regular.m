%
% superclass for all strictly monotonically increasing sequences with
% regular spacing
%
% author: Martin F. Schiffner
% date: 2019-03-29
% modified: 2019-06-08
%
classdef sequence_increasing_regular < math.sequence_increasing

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        q_lb ( 1, 1 ) int64     % lower integer bound
        q_ub ( 1, 1 ) int64     % upper integer bound
        delta ( 1, 1 ) physical_values.physical_quantity       % step size

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = sequence_increasing_regular( lbs_q, ubs_q, deltas )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure integers for lbs_q and ubs_q
            mustBeInteger( lbs_q );
            mustBeInteger( ubs_q );

            % ensure positive deltas
            % (superclass ensures class physical_values.physical_quantity)
            mustBePositive( deltas );

            % multiple lbs_q / single deltas
            if ~isscalar( lbs_q ) && isscalar( deltas )
                deltas = repmat( deltas, size( lbs_q ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( lbs_q, ubs_q, deltas );

            %--------------------------------------------------------------
            % 2.) compute strictly monotonically increasing members
            %--------------------------------------------------------------
            members = cell( size( lbs_q ) );
            for index_object = 1:numel( lbs_q )
                % use double function to prevent zeros caused by int64
                members{ index_object } = double( lbs_q( index_object ):ubs_q( index_object ) )' * deltas( index_object );
            end

            %--------------------------------------------------------------
            % 3.) constructor of superclass
            %--------------------------------------------------------------
            objects@math.sequence_increasing( members );

            %--------------------------------------------------------------
            % 4.) set independent properties
            %--------------------------------------------------------------
            for index_object = 1:numel( objects )
                objects( index_object ).q_lb = int64( lbs_q( index_object ) );
                objects( index_object ).q_ub = int64( ubs_q( index_object ) );
                objects( index_object ).delta = deltas( index_object );
            end

        end % function objects = sequence_increasing_regular( lbs_q, ubs_q, deltas )

        %------------------------------------------------------------------
        % subsample
        %------------------------------------------------------------------
%         function sequences_out = subsample( sequences_in, indices )
% 
%             %--------------------------------------------------------------
%             % 1.) check arguments
%             %--------------------------------------------------------------
%             % ensure class math.sequence_increasing_regular
%             if ~isa( sequences_in, 'math.sequence_increasing_regular' )
%                 errorStruct.message = 'sequences_in must be math.sequence_increasing_regular!';
%                 errorStruct.identifier = 'subsample:NoIncreasingSequences';
%                 error( errorStruct );
%             end
% 
%             %--------------------------------------------------------------
%             % 2.) perform subsampling
%             %--------------------------------------------------------------
%             % specify cell array for sequences_out
%             sequences_out = cell( size( sequences_in ) );
% 
%             % iterate sequences
%             for index_object = 1:numel( sequences_in )
% 
%             % call method subsample in superclass
%             subsample@math.sequence_increasing( sequences_in, indices )
% 
%             
%             % extract cardinalities
%             N_members = abs( sequences_in );
% 
%             % specify cell array for members_sub
%             members_sub = cell( size( sequences_in ) );
% 
%             
% 
%                 % ensure positive integers
%                 mustBeInteger( indices{ index_object } );
%                 mustBePositive( indices{ index_object } );
% 
%                 % ensure that indices{ index_object } do not exceed N_members
%                 if any( indices{ index_object } > N_members( index_object ) )
%                     errorStruct.message = sprintf( 'indices{ %d } must not exceed %d!', index_object, N_members( index_object ) );
%                     errorStruct.identifier = 'subsample:InvalidIndices';
%                     error( errorStruct );
%                 end
% 
%                 % subsample members
%                 members_sub{ index_object } = sequences_in( index_object ).members( indices{ index_object } );
% 
%             end % for index_object = 1:numel( sequences_in )
% 
%             %--------------------------------------------------------------
%             % 3.) create sequences
%             %--------------------------------------------------------------
%             sequences_out = math.sequence_increasing( members_sub );
% 
%         end % function sequences_out = subsample( sequences_in, indices )

        %------------------------------------------------------------------
        % cut out subsequence
        %------------------------------------------------------------------
        function [ sequences, indicators ] = cut_out( sequences, lbs, ubs )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % superclass method cut_out ensures correct arguments

            %--------------------------------------------------------------
            % 2.) perform cut out
            %--------------------------------------------------------------
            % call cut out method in superclass
            [ sequences, indicators ] = cut_out@math.sequence_increasing( sequences, lbs, ubs );

            % ensure cell array for indicators
            if ~iscell( indicators )
                indicators = { indicators };
            end

            % iterate sequences
            for index_object = 1:numel( sequences )

                % find indices and values of nonzero elements
                indices = find( indicators{ index_object } );

                % update lower and upper integer bounds
                sequences( index_object ).q_lb = sequences( index_object ).q_lb + indices( 1 ) - 1;
                sequences( index_object ).q_ub = sequences( index_object ).q_lb + numel( indices ) - 1;

            end % for index_object = 1:numel( sequences )

            % avoid cell array for single sequence
            if isscalar( sequences )
                indicators = indicators{ 1 };
            end

        end % function [ sequences, indicators ] = cut_out( sequences, lbs, ubs )

        %------------------------------------------------------------------
        % remove last members
        %------------------------------------------------------------------
        function [ sequences, N_remove ] = remove_last( sequences, varargin )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % superclass method remove_last ensures correct arguments

            %--------------------------------------------------------------
            % 2.) remove last members
            %--------------------------------------------------------------
            % call remove_last method in superclass
            [ sequences, N_remove ] = remove_last@math.sequence_increasing( sequences, varargin{ : } );

            % iterate sequences
            for index_object = 1:numel( sequences )

                % update upper integer bounds
                sequences( index_object ).q_ub = sequences( index_object ).q_ub - N_remove( index_object );

            end % for index_object = 1:numel( sequences )

        end % function [ sequences, N_remove ] = remove_last( sequences, varargin )

    end % methods

end % classdef sequence_increasing_regular < math.sequence_increasing
